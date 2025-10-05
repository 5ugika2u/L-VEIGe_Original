# modules/enhanced_candidate_gen.py

import pandas as pd
import Levenshtein
import random
import logging
from typing import Dict, List, Optional, Tuple, Set
from database.db_manager import DatabaseManager

class EnhancedCandidateGenerator:
    """
    データベース連携対応の選択肢生成クラス
    既存の選択肢を優先的に再利用し、必要に応じて新規生成する
    """
    
    def __init__(self, db_manager: DatabaseManager, 
                 vocab_path: str = "data/coco_cefr_vocab.csv"):
        """
        EnhancedCandidateGeneratorを初期化
        
        Args:
            db_manager (DatabaseManager): データベース管理インスタンス
            vocab_path (str): 語彙CSVファイルのパス
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # 語彙データの読み込み
        self.coco_vocab = self._load_vocabulary(vocab_path)
        self.logger.info(f"Loaded {len(self.coco_vocab)} vocabulary entries for candidate generation")
        
        # キャッシュ用の辞書
        self._choices_cache = {}
        
        # 選択肢生成のパラメータ
        self.num_distractors = 2  # 誤答選択肢の数
        self.total_choices = 3    # 総選択肢数（正解1 + 誤答2）
        
        # レーベンシュタイン距離による類似度の重み
        self.similarity_weight = 0.7
        self.randomness_weight = 0.3
    
    def _load_vocabulary(self, vocab_path: str) -> pd.DataFrame:
        """
        語彙CSVファイルを読み込み
        
        Args:
            vocab_path (str): 語彙CSVファイルのパス
            
        Returns:
            pd.DataFrame: 語彙データフレーム
        """
        try:
            vocab_df = pd.read_csv(vocab_path)
            required_columns = ['POS', 'CEFR', 'Word']
            
            for col in required_columns:
                if col not in vocab_df.columns:
                    raise ValueError(f"Required column '{col}' not found in vocabulary file")
            
            return vocab_df
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary file: {e}")
            raise
    
    def get_or_generate_choices(self, qid: int, question_data: Dict, 
                               force_regenerate: bool = False) -> List[str]:
        """
        既存の選択肢を取得するか、新しく生成する（メイン関数）
        
        Args:
            qid (int): 問題ID
            question_data (Dict): 問題データ
            force_regenerate (bool): 強制的に再生成するかどうか
            
        Returns:
            List[str]: 選択肢リスト（正解を含む、シャッフル済み）
        """
        # キャッシュキーの生成
        cache_key = f"choices_{qid}"
        
        # 強制再生成でない場合、キャッシュを確認
        if not force_regenerate and cache_key in self._choices_cache:
            self.logger.debug(f"Retrieved choices from cache for QID {qid}")
            return self._choices_cache[cache_key]
        
        # 1. まずデータベースから既存選択肢を検索
        if not force_regenerate:
            existing_choices = self.db_manager.get_choices_by_qid(qid)
            if existing_choices and len(existing_choices) == self.total_choices:
                # 既存の選択肢をシャッフル
                shuffled_choices = existing_choices.copy()
                random.shuffle(shuffled_choices)
                
                # キャッシュに保存
                self._choices_cache[cache_key] = shuffled_choices
                
                self.logger.info(f"Retrieved existing choices for QID {qid}")
                return shuffled_choices
        
        # 2. 既存選択肢がない場合、新規生成
        self.logger.info(f"Generating new choices for QID {qid}")
        new_choices = self._generate_new_choices(question_data)
        
        if new_choices:
            # データベースに保存
            self.db_manager.save_choices(qid, new_choices, question_data['answer'])
            
            # 選択肢をシャッフル
            shuffled_choices = new_choices.copy()
            random.shuffle(shuffled_choices)
            
            # キャッシュに保存
            self._choices_cache[cache_key] = shuffled_choices
            
            self.logger.info(f"Generated and saved new choices for QID {qid}")
            return shuffled_choices
        
        # フォールバック：最低限の選択肢を生成
        self.logger.warning(f"Failed to generate quality choices for QID {qid}, using fallback")
        return self._generate_fallback_choices(question_data)
    
    def _generate_new_choices(self, question_data: Dict) -> List[str]:
        """
        新しい選択肢を生成（既存のadd_candidates_to_question関数を改良）
        
        Args:
            question_data (Dict): 問題データ
            
        Returns:
            List[str]: 新規生成された選択肢リスト
        """
        lemma = question_data.get("lemma")
        pos = question_data.get("pos", "").lower()
        cefr = question_data.get("cefr", "").upper()
        correct_answer = question_data.get("answer")
        
        if not all([lemma, pos, cefr, correct_answer]):
            self.logger.error(f"Missing required data for choice generation: {question_data}")
            return None
        
        # 複数の手法で候補を生成
        candidates = []
        
        # 1. レーベンシュタイン距離による類似語候補
        similarity_candidates = self._get_similarity_candidates(lemma, pos, cefr)
        candidates.extend(similarity_candidates)
        
        # 2. 同じCEFRレベル・品詞からのランダム候補
        random_candidates = self._get_random_candidates(lemma, pos, cefr)
        candidates.extend(random_candidates)
        
        # 3. CEFRレベルを少し緩めた候補（より豊富な選択肢のため）
        relaxed_candidates = self._get_relaxed_cefr_candidates(lemma, pos, cefr)
        candidates.extend(relaxed_candidates)
        
        # 重複除去と正解除外
        unique_candidates = list(set(candidates))
        unique_candidates = [c for c in unique_candidates if c.lower() != correct_answer.lower()]
        
        # 最適な誤答を選択
        best_distractors = self._select_best_distractors(
            lemma, unique_candidates, self.num_distractors
        )
        
        if len(best_distractors) < self.num_distractors:
            self.logger.warning(f"Only found {len(best_distractors)} distractors for {lemma}")
            # 不足分をフォールバック候補で補完
            additional_candidates = self._get_fallback_distractors(lemma, pos, cefr)
            best_distractors.extend(additional_candidates[:self.num_distractors - len(best_distractors)])
        
        # 正解と誤答を組み合わせ
        choices = [correct_answer] + best_distractors[:self.num_distractors]
        
        return choices
    
    def _get_similarity_candidates(self, lemma: str, pos: str, cefr: str, 
                                  max_candidates: int = 10) -> List[str]:
        """
        レーベンシュタイン距離に基づく類似語候補を取得
        
        Args:
            lemma (str): 見出し語
            pos (str): 品詞
            cefr (str): CEFRレベル
            max_candidates (int): 最大候補数
            
        Returns:
            List[str]: 類似語候補リスト
        """
        # 同じ品詞・CEFRレベルの語彙を抽出
        filtered_vocab = self.coco_vocab[
            (self.coco_vocab["POS"].str.lower() == pos.lower()) &
            (self.coco_vocab["CEFR"].str.upper() == cefr.upper()) &
            (self.coco_vocab["Word"].str.lower() != lemma.lower())
        ]["Word"].drop_duplicates().tolist()
        
        if not filtered_vocab:
            return []
        
        # レーベンシュタイン距離を計算
        distances = []
        for word in filtered_vocab:
            distance = Levenshtein.distance(lemma.lower(), word.lower())
            # 距離が小さすぎる（類似しすぎ）場合や大きすぎる場合は除外
            if 1 <= distance <= min(len(lemma), len(word)) // 2 + 2:
                distances.append((word, distance))
        
        # 距離順でソートして上位を選択
        distances.sort(key=lambda x: x[1])
        similarity_candidates = [word for word, _ in distances[:max_candidates]]
        
        self.logger.debug(f"Found {len(similarity_candidates)} similarity candidates for {lemma}")
        return similarity_candidates
    
    def _get_random_candidates(self, lemma: str, pos: str, cefr: str, 
                              max_candidates: int = 15) -> List[str]:
        """
        同じ品詞・CEFRレベルからランダムに候補を取得
        
        Args:
            lemma (str): 見出し語
            pos (str): 品詞
            cefr (str): CEFRレベル
            max_candidates (int): 最大候補数
            
        Returns:
            List[str]: ランダム候補リスト
        """
        filtered_vocab = self.coco_vocab[
            (self.coco_vocab["POS"].str.lower() == pos.lower()) &
            (self.coco_vocab["CEFR"].str.upper() == cefr.upper()) &
            (self.coco_vocab["Word"].str.lower() != lemma.lower())
        ]["Word"].drop_duplicates().tolist()
        
        if not filtered_vocab:
            return []
        
        # ランダムサンプリング
        sample_size = min(max_candidates, len(filtered_vocab))
        random_candidates = random.sample(filtered_vocab, sample_size)
        
        self.logger.debug(f"Found {len(random_candidates)} random candidates for {lemma}")
        return random_candidates
    
    def _get_relaxed_cefr_candidates(self, lemma: str, pos: str, cefr: str, 
                                   max_candidates: int = 10) -> List[str]:
        """
        CEFRレベルを緩めた候補を取得（隣接レベル）
        
        Args:
            lemma (str): 見出し語
            pos (str): 品詞
            cefr (str): CEFRレベル
            max_candidates (int): 最大候補数
            
        Returns:
            List[str]: 緩和候補リスト
        """
        # CEFRレベルの隣接レベルを定義
        cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        try:
            current_index = cefr_levels.index(cefr.upper())
            adjacent_levels = []
            
            # 前後のレベルを追加
            if current_index > 0:
                adjacent_levels.append(cefr_levels[current_index - 1])
            if current_index < len(cefr_levels) - 1:
                adjacent_levels.append(cefr_levels[current_index + 1])
                
        except ValueError:
            # 不明なCEFRレベルの場合は隣接レベルなし
            adjacent_levels = []
        
        relaxed_candidates = []
        for adj_cefr in adjacent_levels:
            candidates = self.coco_vocab[
                (self.coco_vocab["POS"].str.lower() == pos.lower()) &
                (self.coco_vocab["CEFR"].str.upper() == adj_cefr) &
                (self.coco_vocab["Word"].str.lower() != lemma.lower())
            ]["Word"].drop_duplicates().tolist()
            
            if candidates:
                sample_size = min(max_candidates // len(adjacent_levels), len(candidates))
                relaxed_candidates.extend(random.sample(candidates, sample_size))
        
        self.logger.debug(f"Found {len(relaxed_candidates)} relaxed CEFR candidates for {lemma}")
        return relaxed_candidates
    
    def _select_best_distractors(self, target_lemma: str, candidates: List[str], 
                               num_select: int) -> List[str]:
        """
        最適な誤答選択肢を選択
        
        Args:
            target_lemma (str): 対象見出し語
            candidates (List[str]): 候補リスト
            num_select (int): 選択する数
            
        Returns:
            List[str]: 最適な誤答選択肢
        """
        if len(candidates) <= num_select:
            return candidates
        
        # 各候補にスコアを付与
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_distractor_score(target_lemma, candidate)
            scored_candidates.append((candidate, score))
        
        # スコア順でソート（高いスコアが良い誤答）
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 上位を選択（ランダム性も加味）
        top_candidates = scored_candidates[:num_select * 2]  # 上位の2倍から選択
        selected = random.sample(top_candidates, min(num_select, len(top_candidates)))
        
        return [candidate for candidate, _ in selected]
    
    def _calculate_distractor_score(self, target: str, candidate: str) -> float:
        """
        誤答候補のスコアを計算
        
        Args:
            target (str): 対象語
            candidate (str): 候補語
            
        Returns:
            float: スコア（高いほど良い誤答）
        """
        # レーベンシュタイン距離ベースのスコア
        distance = Levenshtein.distance(target.lower(), candidate.lower())
        max_len = max(len(target), len(candidate))
        
        # 適度な類似度を持つものを高く評価
        # 距離が短すぎる（類似しすぎ）や長すぎる（関連性なし）は低く評価
        if max_len == 0:
            similarity_score = 0
        else:
            normalized_distance = distance / max_len
            # 0.3-0.7の範囲で最高スコア
            if 0.3 <= normalized_distance <= 0.7:
                similarity_score = 1.0
            elif normalized_distance < 0.3:
                similarity_score = normalized_distance / 0.3
            else:
                similarity_score = max(0, 1.0 - (normalized_distance - 0.7) / 0.3)
        
        # 長さの類似性スコア
        length_diff = abs(len(target) - len(candidate))
        length_score = max(0, 1.0 - length_diff / max(len(target), len(candidate)))
        
        # 総合スコア
        total_score = (
            self.similarity_weight * similarity_score +
            (1 - self.similarity_weight) * length_score
        )
        
        return total_score
    
    def _get_fallback_distractors(self, lemma: str, pos: str, cefr: str) -> List[str]:
        """
        フォールバック用の誤答候補を取得（品詞のみ一致）
        
        Args:
            lemma (str): 見出し語
            pos (str): 品詞
            cefr (str): CEFRレベル
            
        Returns:
            List[str]: フォールバック候補リスト
        """
        # 品詞のみ一致する語彙を取得（CEFRレベル無視）
        fallback_vocab = self.coco_vocab[
            (self.coco_vocab["POS"].str.lower() == pos.lower()) &
            (self.coco_vocab["Word"].str.lower() != lemma.lower())
        ]["Word"].drop_duplicates().tolist()
        
        if fallback_vocab:
            # ランダムに選択
            sample_size = min(5, len(fallback_vocab))
            return random.sample(fallback_vocab, sample_size)
        
        return []
    
    def _generate_fallback_choices(self, question_data: Dict) -> List[str]:
        """
        最後の手段としてのフォールバック選択肢生成
        
        Args:
            question_data (Dict): 問題データ
            
        Returns:
            List[str]: フォールバック選択肢リスト
        """
        correct_answer = question_data.get("answer", "unknown")
        pos = question_data.get("pos", "").lower()
        
        # シンプルな固定候補リスト（品詞別）
        fallback_dict = {
            "noun": ["thing", "person", "place", "time", "way"],
            "verb": ["go", "come", "make", "take", "get"],
            "adjective": ["good", "new", "big", "small", "old"],
            "adverb": ["well", "now", "here", "there", "very"]
        }
        
        fallback_candidates = fallback_dict.get(pos, ["option1", "option2"])
        
        # 正解を除外
        distractors = [c for c in fallback_candidates if c.lower() != correct_answer.lower()][:2]
        
        choices = [correct_answer] + distractors
        random.shuffle(choices)
        
        self.logger.warning(f"Used fallback choices for {question_data.get('lemma', 'unknown')}")
        return choices
    
    def add_candidates_to_question(self, question: Dict, return_debug: bool = False) -> Dict:
        """
        既存のadd_candidates_to_question関数との互換性維持
        
        Args:
            question (Dict): 問題辞書
            return_debug (bool): デバッグ情報を返すかどうか
            
        Returns:
            Dict: 更新された問題辞書
            (Optional[int]): デバッグ情報（候補数）
        """
        lemma = question.get("lemma")
        pos = question.get("pos", "").lower()
        cefr = question.get("cefr", "").upper()
        
        if not lemma or not cefr:
            question["candidate"] = []
            return (question, 0) if return_debug else question
        
        # 新規生成（qidが不明な場合）
        choices = self._generate_new_choices(question)
        if choices:
            # 正解を除いて候補リストを作成
            candidates = [c for c in choices if c != question.get("answer")]
            question["candidate"] = candidates[:3]  # 最大3個
        else:
            question["candidate"] = []
        
        candidate_count = len(question["candidate"])
        return (question, candidate_count) if return_debug else question
    
    def get_choice_statistics(self, qid: int) -> Optional[Dict]:
        """
        選択肢の統計情報を取得
        
        Args:
            qid (int): 問題ID
            
        Returns:
            Optional[Dict]: 統計情報
        """
        choices = self.db_manager.get_choices_by_qid(qid)
        if not choices:
            return None
        
        correct_answer = self.db_manager.get_correct_answer(qid)
        if not correct_answer:
            return None
        
        # レーベンシュタイン距離の分析
        distances = []
        for choice in choices:
            if choice != correct_answer:
                distance = Levenshtein.distance(correct_answer.lower(), choice.lower())
                distances.append(distance)
        
        return {
            'total_choices': len(choices),
            'correct_answer': correct_answer,
            'avg_distance': sum(distances) / len(distances) if distances else 0,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0,
            'choices': choices
        }
    
    def regenerate_choices_for_question(self, qid: int, question_data: Dict) -> List[str]:
        """
        特定の問題の選択肢を再生成
        
        Args:
            qid (int): 問題ID
            question_data (Dict): 問題データ
            
        Returns:
            List[str]: 新しい選択肢リスト
        """
        return self.get_or_generate_choices(qid, question_data, force_regenerate=True)
    
    def clear_cache(self) -> None:
        """
        選択肢生成キャッシュをクリア
        """
        self._choices_cache.clear()
        self.logger.info("Choice generation cache cleared")
    
    def get_vocabulary_stats(self) -> Dict:
        """
        語彙データの統計情報を取得（選択肢生成の観点から）
        
        Returns:
            Dict: 統計情報
        """
        total_vocab = len(self.coco_vocab)
        
        # 品詞別・CEFRレベル別の語彙数
        stats_by_criteria = {}
        grouped = self.coco_vocab.groupby(['POS', 'CEFR']).size()
        
        for (pos, cefr), count in grouped.items():
            if pos.lower() not in stats_by_criteria:
                stats_by_criteria[pos.lower()] = {}
            stats_by_criteria[pos.lower()][cefr.upper()] = count
        
        return {
            'total_vocabulary': total_vocab,
            'by_pos_cefr': stats_by_criteria,
            'unique_pos': self.coco_vocab['POS'].nunique(),
            'unique_cefr': self.coco_vocab['CEFR'].nunique()
        }


# === 使用例とテスト用コード ===
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # テスト用データベース
    from database.db_manager import DatabaseManager
    
    # DatabaseManagerとEnhancedCandidateGeneratorの初期化
    db_manager = DatabaseManager("test_enhanced_candidates.db")
    candidate_gen = EnhancedCandidateGenerator(db_manager)
    
    print("=== EnhancedCandidateGenerator Test ===")
    
    # 1. 語彙統計の確認
    print("\n1. Vocabulary statistics:")
    stats = candidate_gen.get_vocabulary_stats()
    print(f"  Total vocabulary: {stats['total_vocabulary']}")
    print(f"  Unique POS: {stats['unique_pos']}")
    print(f"  Unique CEFR: {stats['unique_cefr']}")
    
    # 2. テスト問題データの作成
    test_questions = [
        {
            "qid": 1,
            "lemma": "cat",
            "pos": "noun",
            "cefr": "A1",
            "answer": "cat",
            "caption": "A cat sitting on the table"
        },
        {
            "qid": 2,
            "lemma": "run",
            "pos": "verb",
            "cefr": "A2",
            "answer": "running",
            "caption": "A person running in the park"
        },
        {
            "qid": 3,
            "lemma": "beautiful",
            "pos": "adjective",
            "cefr": "B1",
            "answer": "beautiful",
            "caption": "A beautiful flower in the garden"
        }
    ]
    
    # 3. 選択肢生成テスト
    print("\n2. Choice generation test:")
    
    for question in test_questions:
        print(f"\n  Testing question QID {question['qid']} ({question['lemma']}):")
        
        # 初回生成（新規作成）
        choices1 = candidate_gen.get_or_generate_choices(question['qid'], question)
        print(f"    Generated choices: {choices1}")
        
        # 2回目生成（既存取得）
        choices2 = candidate_gen.get_or_generate_choices(question['qid'], question)
        print(f"    Retrieved choices: {choices2}")
        
        # 統計情報の取得
        stats = candidate_gen.get_choice_statistics(question['qid'])
        if stats:
            print(f"    Statistics: avg_distance={stats['avg_distance']:.2f}, min={stats['min_distance']}, max={stats['max_distance']}")
        
        # 再生成テスト
        choices3 = candidate_gen.regenerate_choices_for_question(question['qid'], question)
        print(f"    Regenerated choices: {choices3}")
    
    # 4. 互換性テスト（既存の関数）
    print("\n3. Compatibility test (add_candidates_to_question):")
    
    test_question = {
        "lemma": "dog",
        "pos": "noun",
        "cefr": "A1",
        "answer": "dog"
    }
    
    result = candidate_gen.add_candidates_to_question(test_question, return_debug=True)
    question_with_candidates, debug_count = result
    print(f"  Question: {test_question['lemma']}")
    print(f"  Generated candidates: {question_with_candidates.get('candidate', [])}")
    print(f"  Debug count: {debug_count}")
    
    print("\n=== Test completed ===")