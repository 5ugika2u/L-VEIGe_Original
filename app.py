# app.py - VocabularyLearningApp (Flask) メインアプリケーション

from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_from_directory
import logging
import os
import random
from typing import Dict, List, Optional
from datetime import datetime

# 自作モジュールのインポート
from database.db_manager import DatabaseManager
from modules.enhanced_question_gen import EnhancedQuestionGenerator
from modules.enhanced_candidate_gen import EnhancedCandidateGenerator
from modules.enhanced_image_gen import EnhancedImageGenerator
from modules.result_processor import ResultProcessor

class VocabularyLearningApp:
    """
    語彙学習システムのメインFlaskアプリケーション
    """
    
    def __init__(self, config_dict: Dict = None):
        """
        アプリケーションを初期化
        
        Args:
            config_dict (Dict): 設定辞書（テスト用）
        """
        # デフォルト設定を最初に初期化
        self.config = {
            'DATABASE_PATH': 'vocabulary_learning.db',
            'VOCAB_PATH': 'data/coco_cefr_vocab.csv',
            'CAPTION_PATH': 'data/captions_val2017_sample10.json',  # サンプルデータ用
            'QUESTIONS_PER_SESSION': 10,
            'STATIC_FOLDER': 'static',
            'IMAGES_FOLDER': 'static/images',
            'DEBUG': False
        }
        
        # 設定の更新
        if config_dict:
            self.config.update(config_dict)
        
        # Flaskアプリの初期化
        self.app = Flask(__name__, 
                        static_folder=self.config.get('STATIC_FOLDER', 'static'),
                        static_url_path='/static')
        self.app.secret_key = os.getenv('FLASK_SECRET_KEY', 'vocabulary-learning-secret-key-2024')
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # コンポーネントの初期化
        self._initialize_components()
        
        # ルートの設定
        self._setup_routes()
        
        self.logger.info("VocabularyLearningApp initialized successfully")
    
    def _initialize_components(self):
        """
        システムコンポーネントを初期化
        """
        try:
            # データベース管理
            self.db_manager = DatabaseManager(self.config['DATABASE_PATH'])
            
            # 問題生成
            self.question_generator = EnhancedQuestionGenerator(
                self.db_manager,
                self.config['VOCAB_PATH'],
                self.config['CAPTION_PATH']
            )
            
            # 選択肢生成
            self.candidate_generator = EnhancedCandidateGenerator(
                self.db_manager,
                self.config['VOCAB_PATH']
            )
            
            # 画像生成
            self.image_generator = EnhancedImageGenerator(self.db_manager)
            
            # 結果処理
            self.result_processor = ResultProcessor(
                self.db_manager,
                self.image_generator
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_routes(self):
        """
        Flaskルートを設定
        """
        # === 基本ページ ===
        @self.app.route('/')
        def index():
            return self._index()
        
        @self.app.route('/login')
        def login_page():
            return self._login_page()
        
        # === 学習開始 ===
        @self.app.route('/start_learning', methods=['POST'])
        def start_learning():
            return self._start_learning()
        
        # === 問題表示 ===
        @self.app.route('/question')
        def show_question():
            return self._show_question()
        
        # === 回答処理 ===
        @self.app.route('/answer', methods=['POST'])
        def process_answer():
            return self._process_answer()
        
        # === 結果表示 ===
        @self.app.route('/result')
        def show_result():
            return self._show_result()
        
        # === 次の問題へ ===
        @self.app.route('/next_question', methods=['POST'])
        def next_question():
            return self._next_question()
        
        # === セッション完了 ===
        @self.app.route('/session_complete')
        def session_complete():
            return self._session_complete()
        
        # === 画像配信 ===
        @self.app.route('/images/<path:filename>')
        def serve_image(filename):
            return self._serve_image(filename)
        
        # === 静的画像確認 ===
        @self.app.route('/check_image/<image_id>')
        def check_image(image_id):
            return self._check_static_image(image_id)
        
        # === API エンドポイント ===
        @self.app.route('/api/session_status')
        def api_session_status():
            return self._api_session_status()
        
        @self.app.route('/api/user_stats/<username>')
        def api_user_stats(username):
            return self._api_user_stats(username)
        
        # === 管理機能 ===
        @self.app.route('/admin')
        def admin_page():
            return self._admin_page()
        
        @self.app.route('/admin/stats')
        def admin_stats():
            return self._admin_stats()
    
    # === ルートハンドラー実装 ===
    
    def _index(self):
        """
        トップページ
        """
        return redirect(url_for('login_page'))
    
    def _login_page(self):
        """
        ログイン画面を表示
        """
        # 利用可能な品詞とCEFRレベルを取得
        criteria = self.question_generator.get_available_criteria()
        
        return render_template('login.html', criteria=criteria)
    
    def _start_learning(self):
        """
        学習開始処理
        """
        try:
            # フォームデータの取得
            username = request.form.get('username', '').strip()
            mode = request.form.get('mode', 'learning')
            pos = request.form.get('pos', 'noun')
            cefr = request.form.get('cefr', 'A1')
            
            # Validation
            if not username:
                return render_template('login.html', 
                                     error="Please enter a username.",
                                     criteria=self.question_generator.get_available_criteria())
            
            if mode not in ['learning', 'review']:
                mode = 'learning'
            
            # Get or create user
            user_id = self.db_manager.get_or_create_user(username)
            
            # Create learning session
            session_id = self.db_manager.create_learning_session(
                user_id, mode, pos, cefr, self.config['QUESTIONS_PER_SESSION']
            )
            
            # Save session information
            session['session_id'] = session_id
            session['username'] = username
            session['mode'] = mode
            session['pos'] = pos
            session['cefr'] = cefr
            session['user_id'] = user_id
            
            self.logger.info(f"Started {mode} session for user {username}: {session_id}")
            
            return redirect(url_for('show_question'))
            
        except Exception as e:
            self.logger.error(f"Failed to start learning: {e}")
            return render_template('login.html', 
                                 error="Failed to start learning. Please try again.",
                                 criteria=self.question_generator.get_available_criteria())
    
    def _show_question(self):
        """
        問題表示画面
        """
        if 'session_id' not in session:
            return redirect(url_for('login_page'))
        
        try:
            session_id = session['session_id']
            
            # セッション完了チェック
            if self.result_processor.check_session_completion(session_id):
                return redirect(url_for('session_complete'))
            
            # 現在の問題を取得
            current_question = self._get_current_question()
            
            if not current_question:
                return redirect(url_for('session_complete'))
            
            # セッション情報の取得
            session_summary = self.result_processor.get_session_summary(session_id)
            
            return render_template('question.html', 
                                 question=current_question,
                                 session_info=session_summary,
                                 question_number=session_summary['current_question'] + 1)
            
        except Exception as e:
            self.logger.error(f"Failed to show question: {e}")
            return render_template('error.html', error="Failed to display question.")
    
    # def _get_current_question(self) -> Optional[Dict]:
    #     """
    #     現在の問題を取得
    #     """
    #     try:
    #         session_id = session['session_id']
    #         mode = session['mode']
    #         pos = session['pos']
    #         cefr = session['cefr']
    #         user_id = session['user_id']
            
    #         # 回答済み問題のIDを取得
    #         answered_qids = self.db_manager.get_session_questions_answered(session_id)
            
    #         if mode == 'review':
    #             # 復習モード：過去の問題から選択
    #             review_questions = self.db_manager.get_review_questions(user_id, 20)
    #             available_questions = [q for q in review_questions if q['qid'] not in answered_qids]
                
    #             if available_questions:
    #                 question = random.choice(available_questions)
    #             else:
    #                 # 復習問題がない場合は通常の問題を取得
    #                 question = self.question_generator.get_or_generate_question(pos, cefr, answered_qids)
    #         else:
    #             # 学習モード：新しい問題を生成
    #             question = self.question_generator.get_or_generate_question(pos, cefr, answered_qids)
            
    #         if question and 'choices' in question:
    #             # 選択肢をシャッフル
    #             choices = question['choices'].copy()
    #             random.shuffle(choices)
    #             question['shuffled_choices'] = choices
            
    #         return question
            
    #     except Exception as e:
    #         self.logger.error(f"Failed to get current question: {e}")
    #         return None
    # def _get_current_question(self) -> Optional[Dict]:
    #     """
    #     現在の問題を取得（修正版）
    #     """
    #     try:
    #         session_id = session['session_id']
    #         mode = session['mode']
    #         pos = session['pos']
    #         cefr = session['cefr']
    #         user_id = session['user_id']
            
    #         # 回答済み問題のIDを取得
    #         answered_qids = self.db_manager.get_session_questions_answered(session_id)
            
    #         if mode == 'review':
    #             # 復習モード：過去の問題から選択
    #             review_questions = self.db_manager.get_review_questions(user_id, 20)
    #             available_questions = [q for q in review_questions if q['qid'] not in answered_qids]
                
    #             if available_questions:
    #                 question = random.choice(available_questions)
                    
    #                 # 修正点: 復習問題にも選択肢を追加
    #                 choices = self.db_manager.get_choices_by_qid(question['qid'])
    #                 if choices:
    #                     question['choices'] = choices
    #                     question['candidate'] = [c for c in choices if c != question['answer']]
    #                 else:
    #                     # 選択肢がない場合は新規生成
    #                     self.logger.warning(f"No choices found for review question QID {question['qid']}, generating new choices")
    #                     try:
    #                         from modules.enhanced_candidate_gen import EnhancedCandidateGenerator
    #                         candidate_gen = EnhancedCandidateGenerator(self.db_manager)
    #                         new_choices = candidate_gen.get_or_generate_choices(question['qid'], question)
    #                         if new_choices:
    #                             question['choices'] = new_choices
    #                             question['candidate'] = [c for c in new_choices if c != question['answer']]
    #                     except Exception as e:
    #                         self.logger.error(f"Failed to generate choices for review question: {e}")
    #                         # フォールバック: 通常の問題生成に切り替え
    #                         question = self.question_generator.get_or_generate_question(pos, cefr, answered_qids)
    #             else:
    #                 # 復習問題がない場合は通常の問題を取得
    #                 question = self.question_generator.get_or_generate_question(pos, cefr, answered_qids)
    #         else:
    #             # 学習モード：新しい問題を生成
    #             question = self.question_generator.get_or_generate_question(pos, cefr, answered_qids)
            
    #         # 修正点: すべてのモードで選択肢の存在を確認
    #         if question and 'choices' in question:
    #             # 選択肢をシャッフル
    #             choices = question['choices'].copy()
    #             random.shuffle(choices)
    #             question['shuffled_choices'] = choices
    #         elif question:
    #             # 選択肢がない場合のフォールバック処理
    #             self.logger.warning(f"Question QID {question.get('qid', 'unknown')} has no choices, attempting to generate")
    #             try:
    #                 from modules.enhanced_candidate_gen import EnhancedCandidateGenerator
    #                 candidate_gen = EnhancedCandidateGenerator(self.db_manager)
    #                 new_choices = candidate_gen.get_or_generate_choices(question['qid'], question)
    #                 if new_choices:
    #                     question['choices'] = new_choices
    #                     question['candidate'] = [c for c in new_choices if c != question['answer']]
    #                     # シャッフル
    #                     shuffled_choices = new_choices.copy()
    #                     random.shuffle(shuffled_choices)
    #                     question['shuffled_choices'] = shuffled_choices
    #                 else:
    #                     self.logger.error(f"Failed to generate choices for question QID {question['qid']}")
    #                     return None
    #             except Exception as e:
    #                 self.logger.error(f"Error generating choices: {e}")
    #                 return None
            
    #         return question
            
    #     except Exception as e:
    #         self.logger.error(f"Failed to get current question: {e}")
    #         return None
    def _get_current_question(self) -> Optional[Dict]:
        """
        現在の問題を取得（完全ランダム版 - 復習モード対応）
        """
        try:
            session_id = session['session_id']
            mode = session['mode']
            pos = session['pos']
            cefr = session['cefr']
            user_id = session['user_id']
            
            # 回答済み問題のIDを取得
            answered_qids = self.db_manager.get_session_questions_answered(session_id)
            
            if mode == 'review':
                # 復習モード：過去の問題から選択
                review_questions = self.db_manager.get_review_questions(user_id, 20)
                available_questions = [q for q in review_questions if q['qid'] not in answered_qids]
                
                if available_questions:
                    question = random.choice(available_questions)
                    
                    # 修正点: 復習問題にも選択肢を追加
                    choices = self.db_manager.get_choices_by_qid(question['qid'])
                    if choices:
                        question['choices'] = choices
                        question['candidate'] = [c for c in choices if c != question['answer']]
                    else:
                        # 選択肢がない場合は新規生成
                        self.logger.warning(f"No choices found for review question QID {question['qid']}, generating new choices")
                        try:
                            from modules.enhanced_candidate_gen import EnhancedCandidateGenerator
                            candidate_gen = EnhancedCandidateGenerator(self.db_manager)
                            new_choices = candidate_gen.get_or_generate_choices(question['qid'], question)
                            if new_choices:
                                question['choices'] = new_choices
                                question['candidate'] = [c for c in new_choices if c != question['answer']]
                        except Exception as e:
                            self.logger.error(f"Failed to generate choices for review question: {e}")
                            # フォールバック: 通常の問題生成に切り替え
                            question = self.question_generator.get_or_generate_question(
                                pos, cefr, answered_qids, force_new=True
                            )
                else:
                    # 復習問題がない場合は通常のランダム生成
                    question = self.question_generator.get_or_generate_question(
                        pos, cefr, answered_qids, force_new=True  # 常に新規生成
                    )
            else:
                # 学習モード：CSVからランダムに問題を生成
                question = self.question_generator.get_or_generate_question(
                    pos, cefr, answered_qids, force_new=True  # 常に新規生成
                )
            
            # 修正点: すべてのモードで選択肢の存在を確認
            if question and 'choices' in question:
                # 選択肢をシャッフル
                choices = question['choices'].copy()
                random.shuffle(choices)
                question['shuffled_choices'] = choices
            elif question:
                # 選択肢がない場合のフォールバック処理
                self.logger.warning(f"Question QID {question.get('qid', 'unknown')} has no choices, attempting to generate")
                try:
                    from modules.enhanced_candidate_gen import EnhancedCandidateGenerator
                    candidate_gen = EnhancedCandidateGenerator(self.db_manager)
                    new_choices = candidate_gen.get_or_generate_choices(question['qid'], question)
                    if new_choices:
                        question['choices'] = new_choices
                        question['candidate'] = [c for c in new_choices if c != question['answer']]
                        # シャッフル
                        shuffled_choices = new_choices.copy()
                        random.shuffle(shuffled_choices)
                        question['shuffled_choices'] = shuffled_choices
                    else:
                        self.logger.error(f"Failed to generate choices for question QID {question['qid']}")
                        return None
                except Exception as e:
                    self.logger.error(f"Error generating choices: {e}")
                    return None
            
            return question
            
        except Exception as e:
            self.logger.error(f"Failed to get current question: {e}")
            return None
        
    def _process_answer(self):
        """
        回答処理
        """
        if 'session_id' not in session:
            return redirect(url_for('login_page'))
        
        try:
            # フォームデータの取得
            user_answer = request.form.get('choice', '').strip()
            qid = int(request.form.get('qid', 0))
            
            if not user_answer or not qid:
                return redirect(url_for('show_question'))
            
            # 問題データの取得
            question_data = self.question_generator.get_question_by_id(qid)
            if not question_data:
                return redirect(url_for('show_question'))
            
            # 回答処理
            result = self.result_processor.process_user_answer(
                session['session_id'], qid, question_data, user_answer
            )
            
            # 結果をセッションに保存
            session['last_result'] = result
            
            return redirect(url_for('show_result'))
            
        except Exception as e:
            self.logger.error(f"Failed to process answer: {e}")
            return render_template('error.html', error="Failed to process answer.")
    
    def _show_result(self):
        """
        結果表示画面
        """
        if 'session_id' not in session or 'last_result' not in session:
            return redirect(url_for('show_question'))
        
        try:
            result = session['last_result']
            session_id = session['session_id']
            
            # セッション概要の取得
            session_summary = self.result_processor.get_session_summary(session_id)
            
            # セッション完了チェック
            is_session_complete = self.result_processor.check_session_completion(session_id)
            
            return render_template('result.html', 
                                 result=result,
                                 session_info=session_summary,
                                 is_session_complete=is_session_complete)
            
        except Exception as e:
            self.logger.error(f"Failed to show result: {e}")
            return render_template('error.html', error="Failed to display result.")
    
    def _next_question(self):
        """
        次の問題へ遷移
        """
        if 'session_id' not in session:
            return redirect(url_for('login_page'))
        
        # セッション完了チェック
        if self.result_processor.check_session_completion(session['session_id']):
            return redirect(url_for('session_complete'))
        
        # 最後の結果をクリア
        session.pop('last_result', None)
        
        return redirect(url_for('show_question'))
    
    def _session_complete(self):
        """
        セッション完了画面
        """
        if 'session_id' not in session:
            return redirect(url_for('login_page'))
        
        try:
            session_id = session['session_id']
            
            # セッション概要の取得
            session_summary = self.result_processor.get_session_summary(session_id)
            
            # ユーザー統計の取得
            user_stats = self.db_manager.get_user_statistics(session['user_id'])
            
            return render_template('session_complete.html',
                                 session_info=session_summary,
                                 user_stats=user_stats,
                                 username=session.get('username', 'Unknown'))
            
        except Exception as e:
            self.logger.error(f"Failed to show session complete: {e}")
            return render_template('error.html', error="Failed to display completion screen.")
    
    def _serve_image(self, filename):
        """
        生成された画像を配信
        """
        try:
            # セキュリティチェック
            if '..' in filename or filename.startswith('/'):
                return "Invalid filename", 400
            
            # 画像ディレクトリから配信
            return send_from_directory(self.image_generator.base_output_dir, filename)
            
        except Exception as e:
            self.logger.error(f"Failed to serve image {filename}: {e}")
            return "Image not found", 404
    
    def _check_static_image(self, image_id):
        """
        静的画像の存在確認
        """
        try:
            # 画像ファイル名の生成
            filename = f"{str(image_id).zfill(12)}.jpg"
            image_path = os.path.join(self.config['IMAGES_FOLDER'], filename)
            
            if os.path.exists(image_path):
                return jsonify({'exists': True, 'path': f"/static/images/{filename}"})
            else:
                return jsonify({'exists': False, 'placeholder': "/static/placeholder.jpg"})
                
        except Exception as e:
            self.logger.error(f"Failed to check image {image_id}: {e}")
            return jsonify({'exists': False, 'error': str(e)})
    
    # === API エンドポイント ===
    
    def _api_session_status(self):
        """
        セッション状態API
        """
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
        
        try:
            session_summary = self.result_processor.get_session_summary(session['session_id'])
            return jsonify(session_summary)
            
        except Exception as e:
            self.logger.error(f"API session status error: {e}")
            return jsonify({'error': 'Failed to get session status'}), 500
    
    def _api_user_stats(self, username):
        """
        ユーザー統計API
        """
        try:
            user_id = self.db_manager.get_or_create_user(username)
            stats = self.db_manager.get_user_statistics(user_id)
            return jsonify(stats)
            
        except Exception as e:
            self.logger.error(f"API user stats error: {e}")
            return jsonify({'error': 'Failed to get user statistics'}), 500
    
    # === 管理機能 ===
    
    def _admin_page(self):
        """
        管理画面
        """
        # 簡易認証（本番環境では適切な認証を実装）
        if not self.config.get('DEBUG', False):
            return "Admin access disabled", 403
        
        return render_template('admin.html')
    
    def _admin_stats(self):
        """
        システム統計API
        """
        if not self.config.get('DEBUG', False):
            return jsonify({'error': 'Admin access disabled'}), 403
        
        try:
            stats = {
                'question_stats': self.question_generator.get_vocabulary_stats(),
                'candidate_stats': self.candidate_generator.get_vocabulary_stats(),
                'image_stats': self.image_generator.get_image_generation_stats(),
                'system_info': {
                    'database_path': self.config['DATABASE_PATH'],
                    'questions_per_session': self.config['QUESTIONS_PER_SESSION']
                }
            }
            return jsonify(stats)
            
        except Exception as e:
            self.logger.error(f"Admin stats error: {e}")
            return jsonify({'error': 'Failed to get system statistics'}), 500
    
    # === ユーティリティメソッド ===
    
    def clear_session(self):
        """
        セッションをクリア
        """
        session.clear()
    
    def get_app(self):
        """
        Flaskアプリインスタンスを取得
        """
        return self.app
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """
        アプリケーションを実行
        """
        self.app.run(host=host, port=port, debug=debug)


# === アプリケーション作成関数 ===
def create_app(config_dict: Dict = None) -> Flask:
    """
    Flaskアプリケーションを作成する関数
    
    Args:
        config_dict (Dict): 設定辞書
        
    Returns:
        Flask: 設定済みFlaskアプリケーション
    """
    vocab_app = VocabularyLearningApp(config_dict)
    return vocab_app.get_app()


# === メイン実行部 ===
if __name__ == '__main__':
    # 開発用設定
    config = {
        'DEBUG': True,
        'QUESTIONS_PER_SESSION': 5  # デバッグ用に少なく設定
    }
    
    # アプリケーションの作成と実行
    vocab_app = VocabularyLearningApp(config)
    
    print("=== Vocabulary Learning System ===")
    print("Starting Flask application...")
    print("Access: http://127.0.0.1:5000")
    print("Ctrl+C to stop")
    
    vocab_app.run(debug=True)