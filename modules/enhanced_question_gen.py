# modules/enhanced_question_gen.py - 完全ランダム問題生成版

import pandas as pd
import random
import json
import spacy
import logging
from typing import Dict, List, Optional, Set
from database.db_manager import DatabaseManager

class EnhancedQuestionGenerator:
    """
    データベース連携対応の問題生成クラス
    CSVからランダムに問題を生成し、必要に応じて既存問題を再利用する
    """
    
    def __init__(self, db_manager: DatabaseManager, 
                 vocab_path: str = "data/coco_cefr_vocab.csv",
                 caption_path: str = "data/captions_val2017.json"):
        """
        EnhancedQuestionGeneratorを初期化
        
        Args:
            db_manager (DatabaseManager): データベース管理インスタンス
            vocab_path (str): 語彙CSVファイルのパス
            caption_path (str): COCOキャプションJSONファイルのパス
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # SpaCyモデルの読み込み
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("SpaCy model loaded successfully")
        except OSError:
            raise RuntimeError("Please install spaCy model: run `python -m spacy download en_core_web_sm`")
        
        # 語彙データの読み込み
        self.coco_vocab = self._load_vocabulary(vocab_path)
        self.logger.info(f"Loaded {len(self.coco_vocab)} vocabulary entries")
        
        # キャプションデータの読み込み
        self.caption_dict = self._load_captions(caption_path)
        self.logger.info(f"Loaded {len(self.caption_dict)} captions")
        
        # キャッシュ用の辞書
        self._question_cache = {}
    
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
            required_columns = ['POS', 'CEFR', 'Word', 'CaptionID', 'ImageID']
            
            for col in required_columns:
                if col not in vocab_df.columns:
                    raise ValueError(f"Required column '{col}' not found in vocabulary file")
            
            return vocab_df
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary file: {e}")
            raise
    
    def _load_captions(self, caption_path: str) -> Dict[str, str]:
        """
        COCOキャプションJSONファイルを読み込み
        
        Args:
            caption_path (str): キャプションJSONファイルのパス
            
        Returns:
            Dict[str, str]: キャプションID → キャプションテキストの辞書
        """
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                caption_json = json.load(f)
            
            caption_dict = {}
            for item in caption_json["annotations"]:
                caption_dict[str(item["id"])] = item["caption"]
            
            return caption_dict
        except Exception as e:
            self.logger.error(f"Failed to load caption file: {e}")
            raise
    
    def get_or_generate_question(self, pos_filter: str, cefr_filter: str, 
                               exclude_qids: List[int] = None, 
                               force_new: bool = True,
                               max_attempts: int = 200) -> Optional[Dict]:
        """
        完全ランダム問題生成版
        
        Args:
            pos_filter (str): 品詞フィルター 
            cefr_filter (str): CEFRレベルフィルター 
            exclude_qids (List[int]): 除外する問題IDリスト
            force_new (bool): 強制的に新規生成するかどうか（デフォルト: True）
            max_attempts (int): 新規生成時の最大試行回数
            
        Returns:
            Optional[Dict]: 問題データ（qidを含む）
        """
        
        # 完全ランダム問題生成を優先（CSVからランダム選択）
        self.logger.info(f"Generating random question for {pos_filter} {cefr_filter}")
        new_question = self._generate_random_question_from_csv(pos_filter, cefr_filter, exclude_qids, max_attempts)
        
        if new_question:
            # データベースに既存かチェック
            existing_qid = self._check_existing_question(new_question)
            
            if existing_qid:
                # 既存問題の場合、qidを設定して選択肢を取得
                new_question['qid'] = existing_qid
                choices = self.db_manager.get_choices_by_qid(existing_qid)
                if choices:
                    new_question['choices'] = choices
                    new_question['candidate'] = [c for c in choices if c != new_question['answer']]
                    self.logger.info(f"Reusing existing question: QID {existing_qid}, lemma: {new_question['lemma']}")
            else:
                # 新規問題の場合、データベースに保存
                qid = self.db_manager.save_question(new_question)
                new_question['qid'] = qid
                
                # 選択肢を生成・保存
                self._generate_and_save_choices(new_question)
                self.logger.info(f"Created new question: QID {qid}, lemma: {new_question['lemma']}")
        
        return new_question

    def _generate_random_question_from_csv(self, pos_filter: str, cefr_filter: str, 
                                         exclude_qids: List[int] = None, 
                                         max_attempts: int = 200) -> Optional[Dict]:
        """
        CSVからランダムに問題を生成（既存問題も含む）
        
        Args:
            pos_filter (str): 品詞フィルター
            cefr_filter (str): CEFRレベルフィルター
            exclude_qids (List[int]): 除外する問題IDリスト
            max_attempts (int): 最大試行回数
            
        Returns:
            Optional[Dict]: 新規生成された問題データ
        """
        # セッション内で回答済みの語彙を取得
        excluded_lemmas = self._get_excluded_lemmas_from_session(exclude_qids)
        
        # 条件に合う語彙をフィルタリング
        filtered_vocab = self.coco_vocab[
            (self.coco_vocab["POS"].str.lower() == pos_filter.lower()) &
            (self.coco_vocab["CEFR"].str.upper() == cefr_filter.upper()) &
            (~self.coco_vocab["Word"].isin(excluded_lemmas))
        ]
        
        if filtered_vocab.empty:
            self.logger.warning(f"No vocabulary found for {pos_filter} {cefr_filter}")
            return None
        
        self.logger.info(f"Found {len(filtered_vocab)} candidate vocabularies for {pos_filter} {cefr_filter}")
        
        # 完全ランダムサンプリング（データベースの存在は無視）
        attempts = 0
        max_sample_attempts = min(max_attempts, len(filtered_vocab))
        
        # ランダムに語彙をサンプリング（重複なし）
        sampled_vocab = filtered_vocab.sample(n=max_sample_attempts, replace=False).reset_index(drop=True)
        
        for _, row in sampled_vocab.iterrows():
            attempts += 1
            word = row["Word"]
            
            # キャプション情報の取得
            cap_id = str(row["CaptionID"])
            img_id = str(row["ImageID"])
            original_caption = self.caption_dict.get(cap_id)
            
            if original_caption:
                # SpaCyによる解析
                question_data = self._process_caption_with_spacy(
                    original_caption, word, cap_id, img_id, pos_filter, cefr_filter
                )
                
                if question_data:
                    self.logger.info(f"Successfully generated question for lemma: {word} (attempt {attempts}/{max_sample_attempts})")
                    return question_data
            
            # 進捗ログ
            if attempts % 50 == 0:
                self.logger.info(f"Question generation progress: {attempts}/{max_sample_attempts} attempts")
        
        self.logger.warning(f"Failed to generate question after {attempts} attempts")
        return None

    def _get_excluded_lemmas_from_session(self, exclude_qids: List[int] = None) -> Set[str]:
        """
        セッション内で除外すべき語彙リストを取得（既存問題の語彙は除外しない）
        
        Args:
            exclude_qids (List[int]): セッション内で回答済みの問題IDリスト
            
        Returns:
            Set[str]: 除外する語彙の集合
        """
        excluded_lemmas = set()
        
        if exclude_qids:
            # セッション内で回答済みの語彙のみ除外
            for qid in exclude_qids:
                question = self.db_manager.get_question_by_id(qid)
                if question:
                    excluded_lemmas.add(question['lemma'])
        
        return excluded_lemmas

    def _check_existing_question(self, question_data: Dict) -> Optional[int]:
        """
        同じ語彙・品詞・CEFRの問題がデータベースに存在するかチェック
        
        Args:
            question_data (Dict): 問題データ
            
        Returns:
            Optional[int]: 既存問題のQID（存在しない場合はNone）
        """
        lemma = question_data.get("lemma")
        pos = question_data.get("pos")
        cefr = question_data.get("cefr")
        
        if not all([lemma, pos, cefr]):
            return None
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT qid FROM questions WHERE lemma = ? AND pos = ? AND cefr = ?",
                (lemma, pos.lower(), cefr.upper())
            )
            result = cursor.fetchone()
            return result['qid'] if result else None
    
    def _process_caption_with_spacy(self, caption: str, target_word: str, 
                                   cap_id: str, img_id: str, 
                                   pos_filter: str, cefr_filter: str) -> Optional[Dict]:
        """
        SpaCyを使用してキャプションを処理し、問題データを生成
        
        Args:
            caption (str): 元のキャプション
            target_word (str): 対象語彙
            cap_id (str): キャプションID
            img_id (str): 画像ID
            pos_filter (str): 品詞フィルター
            cefr_filter (str): CEFRレベルフィルター
            
        Returns:
            Optional[Dict]: 問題データ
        """
        try:
            doc = self.nlp(caption)
            
            # divided配列の生成（副詞以外は見出し語に変換）
            divided = []
            for token in doc:
                if token.pos_ != "ADV":
                    divided.append(token.lemma_)
                else:
                    divided.append(token.text)
            
            # 対象語彙を見つけて空所に変換
            answer_word = None
            blanked_tokens = []
            
            for token in doc:
                if token.lemma_.lower() == target_word.lower():
                    answer_word = token.text
                    blanked_tokens.append("()")
                else:
                    blanked_tokens.append(token.text)
            
            # 対象語彙が見つからない場合
            if not answer_word:
                self.logger.debug(f"Target word '{target_word}' not found in caption: {caption}")
                return None
            
            # 問題データの構築
            question_data = {
                "image_id": img_id,
                "id": cap_id,
                "caption": caption,
                "divided": divided,
                "blankquestion": blanked_tokens,
                "answer": answer_word,
                "lemma": target_word,
                "pos": pos_filter.lower(),
                "cefr": cefr_filter.upper()
            }
            
            return question_data
            
        except Exception as e:
            self.logger.error(f"Error processing caption with SpaCy: {e}")
            return None
    
    def _generate_and_save_choices(self, question_data: Dict) -> None:
        """
        選択肢を生成してデータベースに保存
        
        Args:
            question_data (Dict): 問題データ（qidを含む）
        """
        try:
            # EnhancedCandidateGeneratorを使用
            from modules.enhanced_candidate_gen import EnhancedCandidateGenerator
            
            candidate_gen = EnhancedCandidateGenerator(self.db_manager)
            choices = candidate_gen.get_or_generate_choices(question_data['qid'], question_data)
            
            question_data['choices'] = choices
            question_data['candidate'] = [c for c in choices if c != question_data['answer']]
            
            self.logger.info(f"Generated choices using EnhancedCandidateGenerator for QID {question_data['qid']}")
            
        except ImportError:
            # EnhancedCandidateGeneratorがまだない場合の簡易実装
            self.logger.warning("EnhancedCandidateGenerator not available, using simple choice generation")
            choices = self._generate_simple_choices(question_data)
            
            self.db_manager.save_choices(question_data['qid'], choices, question_data['answer'])
            question_data['choices'] = choices
            question_data['candidate'] = [c for c in choices if c != question_data['answer']]
    
    def _generate_simple_choices(self, question_data: Dict) -> List[str]:
        """
        簡易的な選択肢生成（EnhancedCandidateGeneratorが利用できない場合）
        
        Args:
            question_data (Dict): 問題データ
            
        Returns:
            List[str]: 選択肢リスト（正解を含む）
        """
        lemma = question_data["lemma"]
        pos = question_data["pos"]
        cefr = question_data["cefr"]
        
        # 同じ品詞・CEFRレベルの語彙からランダムに選択
        similar_vocab = self.coco_vocab[
            (self.coco_vocab["POS"].str.lower() == pos.lower()) &
            (self.coco_vocab["CEFR"].str.upper() == cefr.upper()) &
            (self.coco_vocab["Word"].str.lower() != lemma.lower())
        ]["Word"].drop_duplicates().tolist()
        
        # ランダムに2つの誤答を選択
        if len(similar_vocab) >= 2:
            wrong_choices = random.sample(similar_vocab, 2)
        else:
            # 十分な語彙がない場合は、CEFRレベルを無視
            fallback_vocab = self.coco_vocab[
                (self.coco_vocab["POS"].str.lower() == pos.lower()) &
                (self.coco_vocab["Word"].str.lower() != lemma.lower())
            ]["Word"].drop_duplicates().tolist()
            
            wrong_choices = random.sample(fallback_vocab, min(2, len(fallback_vocab)))
        
        # 正解と誤答を組み合わせてシャッフル
        choices = [question_data["answer"]] + wrong_choices
        random.shuffle(choices)
        
        return choices
    
    def get_question_by_id(self, qid: int) -> Optional[Dict]:
        """
        問題IDから問題データを取得（選択肢付き）
        
        Args:
            qid (int): 問題ID
            
        Returns:
            Optional[Dict]: 問題データ
        """
        question = self.db_manager.get_question_by_id(qid)
        if question:
            choices = self.db_manager.get_choices_by_qid(qid)
            if choices:
                question['choices'] = choices
                question['candidate'] = [c for c in choices if c != question['answer']]
        
        return question
    
    def get_available_criteria(self) -> Dict[str, List[str]]:
        """
        利用可能な品詞とCEFRレベルの組み合わせを取得
        
        Returns:
            Dict[str, List[str]]: 品詞別のCEFRレベルリスト
        """
        criteria = {}
        grouped = self.coco_vocab.groupby('POS')['CEFR'].unique()
        
        for pos, cefr_levels in grouped.items():
            criteria[pos.lower()] = sorted(cefr_levels.tolist())
        
        return criteria
    
    def get_vocabulary_stats(self) -> Dict:
        """
        語彙データの統計情報を取得
        
        Returns:
            Dict: 統計情報
        """
        total_vocab = len(self.coco_vocab)
        by_pos = self.coco_vocab['POS'].value_counts().to_dict()
        by_cefr = self.coco_vocab['CEFR'].value_counts().to_dict()
        
        return {
            'total_vocabulary': total_vocab,
            'by_pos': by_pos,
            'by_cefr': by_cefr,
            'total_captions': len(self.caption_dict)
        }
    
    def clear_cache(self) -> None:
        """
        問題生成キャッシュをクリア
        """
        self._question_cache.clear()
        self.logger.info("Question generation cache cleared")
    
    def validate_data_integrity(self) -> Dict[str, List[str]]:
        """
        データの整合性をチェック
        
        Returns:
            Dict[str, List[str]]: 検出された問題のリスト
        """
        issues = {
            'missing_captions': [],
            'invalid_vocab_entries': [],
            'spacy_processing_errors': []
        }
        
        # キャプション不整合のチェック
        unique_caption_ids = self.coco_vocab['CaptionID'].unique()
        for cap_id in unique_caption_ids[:100]:  # サンプルチェック
            if str(cap_id) not in self.caption_dict:
                issues['missing_captions'].append(str(cap_id))
        
        # SpaCy処理エラーのチェック
        sample_captions = list(self.caption_dict.values())[:50]  # サンプルチェック
        for caption in sample_captions:
            try:
                self.nlp(caption)
            except Exception as e:
                issues['spacy_processing_errors'].append(f"Caption: {caption[:50]}... Error: {str(e)}")
        
        return issues


# === 使用例とテスト用コード ===
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # テスト用データベース
    from database.db_manager import DatabaseManager
    
    # DatabaseManagerとEnhancedQuestionGeneratorの初期化
    db_manager = DatabaseManager("test_enhanced_questions.db")
    question_gen = EnhancedQuestionGenerator(db_manager)
    
    print("=== EnhancedQuestionGenerator Test ===")
    
    # 1. データ整合性チェック
    print("\n1. Data integrity check:")
    issues = question_gen.validate_data_integrity()
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"  {issue_type}: {len(issue_list)} issues found")
        else:
            print(f"  {issue_type}: OK")
    
    # 2. 利用可能な条件の確認
    print("\n2. Available criteria:")
    criteria = question_gen.get_available_criteria()
    for pos, cefr_levels in criteria.items():
        print(f"  {pos}: {cefr_levels}")
    
    # 3. 語彙統計の確認
    print("\n3. Vocabulary statistics:")
    stats = question_gen.get_vocabulary_stats()
    print(f"  Total vocabulary: {stats['total_vocabulary']}")
    print(f"  Total captions: {stats['total_captions']}")
    
    print("\n=== Test completed ===")