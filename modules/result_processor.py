# modules/result_processor.py

import logging
from typing import Dict, Optional
from database.db_manager import DatabaseManager
from modules.enhanced_image_gen import EnhancedImageGenerator

class ResultProcessor:
    """
    学習者の回答を処理し、適切なフィードバックを生成するクラス
    正答・誤答に応じた画像生成とログ記録を統合管理
    """
    
    def __init__(self, db_manager: DatabaseManager, image_generator: EnhancedImageGenerator):
        """
        ResultProcessorを初期化
        
        Args:
            db_manager (DatabaseManager): データベース管理インスタンス
            image_generator (EnhancedImageGenerator): 画像生成インスタンス
        """
        self.db_manager = db_manager
        self.image_generator = image_generator
        self.logger = logging.getLogger(__name__)
    
    def process_user_answer(self, session_id: str, qid: int, question_data: Dict, 
                           user_answer: str) -> Dict:
        """
        ユーザーの回答を処理し、適切なフィードバックを生成
        
        Args:
            session_id (str): 学習セッションID
            qid (int): 問題ID
            question_data (Dict): 問題データ
            user_answer (str): ユーザーの回答
            
        Returns:
            Dict: 処理結果とフィードバック情報
        """
        # 正解の取得
        correct_answer = question_data.get('answer')
        is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
        
        # セッション情報の取得
        session_info = self.db_manager.get_session_info(session_id)
        if not session_info:
            raise ValueError(f"Session not found: {session_id}")
        
        user_id = session_info['user_id']
        
        # 基本的な結果データ
        result_data = {
            'qid': qid,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'question_data': question_data,
            'session_id': session_id
        }
        
        if is_correct:
            # 正答処理
            feedback = self._process_correct_answer(result_data)
        else:
            # 誤答処理（画像生成を含む）
            feedback = self._process_incorrect_answer(result_data)
        
        # 学習ログの記録
        self._save_learning_log(user_id, session_id, qid, user_answer, is_correct, 
                               feedback.get('generated_image_path'))
        
        # セッション進捗の更新
        self.db_manager.update_session_progress(session_id)
        
        return feedback
    
    def _process_correct_answer(self, result_data: Dict) -> Dict:
        """
        正答時の処理
        
        Args:
            result_data (Dict): 基本結果データ
            
        Returns:
            Dict: 正答フィードバック
        """
        question_data = result_data['question_data']
        
        # 正答文の生成（空所に正解を補完）
        correct_sentence = self._generate_completed_sentence(
            question_data['blankquestion'], 
            result_data['correct_answer']
        )
        
        feedback = {
            'result_type': 'correct',
            'is_correct': True,
            'message': '正解！',
            'completed_sentence': correct_sentence,
            'original_caption': question_data['caption'],
            'correct_answer': result_data['correct_answer'],
            'explanation': f"正解は「{result_data['correct_answer']}」です。",
            'image_path': None,  # 正答時は画像生成なし
            'generated_image_path': None,
            'image_id': question_data.get('image_id')
        }
        
        self.logger.info(f"Processed correct answer for QID {result_data['qid']}")
        return feedback
    
    def _process_incorrect_answer(self, result_data: Dict) -> Dict:
        """
        誤答時の処理（誤答画像生成を含む）
        
        Args:
            result_data (Dict): 基本結果データ
            
        Returns:
            Dict: 誤答フィードバック
        """
        question_data = result_data['question_data']
        user_answer = result_data['user_answer']
        qid = result_data['qid']
        
        # 誤答画像の生成または取得
        try:
            generated_image_path = self.image_generator.get_or_generate_wrong_image(
                qid, question_data, user_answer
            )
        except Exception as e:
            self.logger.error(f"Failed to generate wrong image for QID {qid}: {e}")
            generated_image_path = None
        
        # 誤答文の生成（空所に誤答を補完）
        incorrect_sentence = self._generate_completed_sentence(
            question_data['blankquestion'], 
            user_answer
        )
        
        feedback = {
            'result_type': 'incorrect',
            'is_correct': False,
            'message': '不正解',
            'completed_sentence': incorrect_sentence,
            'original_caption': question_data['caption'],
            'user_answer': user_answer,
            'correct_answer': result_data['correct_answer'],  # 正解は表示しない（仕様通り）
            'show_correct_answer': False,  # 仕様：誤答時は正解を表示しない
            'generated_image_path': generated_image_path,
            'image_available': generated_image_path is not None,
            'explanation': f"あなたの回答「{user_answer}」での場面を画像で確認してください。",
            'image_id': question_data.get('image_id')
        }
        
        self.logger.info(f"Processed incorrect answer for QID {qid}, generated image: {generated_image_path}")
        return feedback
    
    def _generate_completed_sentence(self, blank_question: list, answer: str) -> str:
        """
        空所補充問題に回答を補完して完全な文を生成
        
        Args:
            blank_question (list): 空所を含む問題文のトークンリスト
            answer (str): 補完する答え
            
        Returns:
            str: 完成した文
        """
        completed_tokens = []
        for token in blank_question:
            if token == "()":
                completed_tokens.append(answer)
            else:
                completed_tokens.append(token)
        
        # トークンを文に結合（適切なスペース処理）
        completed_sentence = self._join_tokens_to_sentence(completed_tokens)
        return completed_sentence
    
    def _join_tokens_to_sentence(self, tokens: list) -> str:
        """
        トークンリストを自然な文に結合
        
        Args:
            tokens (list): トークンリスト
            
        Returns:
            str: 結合された文
        """
        if not tokens:
            return ""
        
        sentence = tokens[0]
        
        for i, token in enumerate(tokens[1:], 1):
            # 句読点の前にはスペースを入れない
            if token in ".,!?;:":
                sentence += token
            # アポストロフィの処理
            elif token.startswith("'") or (i > 0 and tokens[i-1].endswith("'")):
                sentence += token
            # 通常の単語の前にはスペースを入れる
            else:
                sentence += " " + token
        
        return sentence
    
    def _save_learning_log(self, user_id: int, session_id: str, qid: int, 
                          selected_choice: str, is_correct: bool, 
                          generated_image_path: Optional[str]) -> None:
        """
        学習ログを保存
        
        Args:
            user_id (int): ユーザーID
            session_id (str): セッションID
            qid (int): 問題ID
            selected_choice (str): 選択した回答
            is_correct (bool): 正答かどうか
            generated_image_path (Optional[str]): 生成された画像のパス
        """
        log_data = {
            'user_id': user_id,
            'qid': qid,
            'selected_choice': selected_choice,
            'is_correct': is_correct,
            'generated_image_path': generated_image_path,
            'session_id': session_id
        }
        
        try:
            self.db_manager.save_learning_log(log_data)
            self.logger.debug(f"Saved learning log for user {user_id}, QID {qid}")
        except Exception as e:
            self.logger.error(f"Failed to save learning log: {e}")
    
    def get_session_summary(self, session_id: str) -> Dict:
        """
        セッションの概要を取得
        
        Args:
            session_id (str): セッションID
            
        Returns:
            Dict: セッション概要
        """
        session_info = self.db_manager.get_session_info(session_id)
        if not session_info:
            return {'error': 'Session not found'}
        
        # セッション内の回答履歴を取得
        answered_qids = self.db_manager.get_session_questions_answered(session_id)
        
        # 統計計算
        correct_count = 0
        total_count = len(answered_qids)
        
        for qid in answered_qids:
            # 各問題の最新の回答を確認
            # 簡易実装：実際にはより詳細な履歴確認が必要
            pass
        
        # 進捗率の計算
        progress_rate = (session_info['current_question'] / session_info['total_questions']) * 100
        
        summary = {
            'session_id': session_id,
            'user_id': session_info['user_id'],
            'mode': session_info['mode'],
            'pos_filter': session_info['pos_filter'],
            'cefr_filter': session_info['cefr_filter'],
            'total_questions': session_info['total_questions'],
            'current_question': session_info['current_question'],
            'answered_questions': total_count,
            'progress_rate': round(progress_rate, 1),
            'is_completed': session_info['is_completed'],
            'answered_qids': answered_qids
        }
        
        return summary
    
    def check_session_completion(self, session_id: str) -> bool:
        """
        セッションが完了しているかチェック
        
        Args:
            session_id (str): セッションID
            
        Returns:
            bool: 完了している場合True
        """
        session_info = self.db_manager.get_session_info(session_id)
        if not session_info:
            return False
        
        # 問題数が上限に達した場合
        if session_info['current_question'] >= session_info['total_questions']:
            # セッションを完了状態に更新
            self.db_manager.complete_session(session_id)
            return True
        
        return session_info['is_completed']
    
    def generate_feedback_message(self, result_data: Dict) -> str:
        """
        結果に応じたフィードバックメッセージを生成
        
        Args:
            result_data (Dict): 結果データ
            
        Returns:
            str: フィードバックメッセージ
        """
        if result_data['is_correct']:
            messages = [
                "素晴らしい！正解です。",
                "よくできました！",
                "その通りです！",
                "正解！よく理解していますね。"
            ]
        else:
            messages = [
                "もう一度考えてみましょう。",
                "画像を見て確認してみてください。",
                "違う答えですが、学習の機会です。",
                "間違いから学ぶことも大切です。"
            ]
        
        # 簡単なランダム選択（実際にはより高度な選択ロジックも可能）
        import random
        return random.choice(messages)


# === 使用例とテスト用コード ===
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # テスト用のコンポーネント初期化
    from database.db_manager import DatabaseManager
    
    db_manager = DatabaseManager("test_result_processor.db")
    image_generator = EnhancedImageGenerator(db_manager)
    result_processor = ResultProcessor(db_manager, image_generator)
    
    print("=== ResultProcessor Test ===")
    
    # テスト用データ
    test_question = {
        'qid': 1,
        'answer': 'cat',
        'caption': 'A cat sitting on the table',
        'blankquestion': ['A', '()', 'sitting', 'on', 'the', 'table'],
        'lemma': 'cat',
        'pos': 'noun',
        'cefr': 'A1'
    }
    
    # テスト用セッション作成
    user_id = db_manager.get_or_create_user("test_user")
    session_id = db_manager.create_learning_session(user_id, "learning", "noun", "A1")
    
    print(f"Created test session: {session_id}")
    print(f"Test question: {test_question}")
    
    # 正答テスト
    print("\n1. Testing correct answer:")
    correct_result = result_processor.process_user_answer(
        session_id, test_question['qid'], test_question, "cat"
    )
    print(f"  Result type: {correct_result['result_type']}")
    print(f"  Message: {correct_result['message']}")
    print(f"  Completed sentence: {correct_result['completed_sentence']}")
    
    # 誤答テスト
    print("\n2. Testing incorrect answer:")
    incorrect_result = result_processor.process_user_answer(
        session_id, test_question['qid'], test_question, "dog"
    )
    print(f"  Result type: {incorrect_result['result_type']}")
    print(f"  Message: {incorrect_result['message']}")
    print(f"  Completed sentence: {incorrect_result['completed_sentence']}")
    print(f"  Image generated: {incorrect_result['image_available']}")
    
    # セッション概要テスト
    print("\n3. Session summary:")
    summary = result_processor.get_session_summary(session_id)
    print(f"  Progress: {summary['current_question']}/{summary['total_questions']}")
    print(f"  Progress rate: {summary['progress_rate']}%")
    print(f"  Answered questions: {summary['answered_questions']}")
    
    # 完了チェックテスト
    print("\n4. Session completion check:")
    is_completed = result_processor.check_session_completion(session_id)
    print(f"  Session completed: {is_completed}")
    
    print("\n=== Test completed ===")