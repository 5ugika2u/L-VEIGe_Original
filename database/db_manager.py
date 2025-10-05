# database/db_manager.py

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import logging

class DatabaseManager:
    """
    語彙学習システムのデータベース管理クラス
    SQLiteを使用してユーザー、問題、学習ログなどを管理する
    """
    
    def __init__(self, db_path: str = "vocabulary_learning.db"):
        """
        DatabaseManagerを初期化
        
        Args:
            db_path (str): データベースファイルのパス
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """
        データベース接続のコンテキストマネージャー
        トランザクションの自動管理を行う
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 結果を辞書形式で取得
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self) -> None:
        """
        データベースとテーブルを初期化
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # ユーザー管理テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 問題管理テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    qid INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    caption_id TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    lemma TEXT NOT NULL,
                    pos TEXT NOT NULL,
                    cefr TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    blank_question TEXT NOT NULL,
                    divided TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lemma, pos, cefr)
                )
            """)
            
            # 選択肢管理テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS choices (
                    choice_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qid INTEGER NOT NULL,
                    choice_text TEXT NOT NULL,
                    is_correct BOOLEAN DEFAULT FALSE,
                    choice_order INTEGER DEFAULT 0,
                    FOREIGN KEY (qid) REFERENCES questions (qid) ON DELETE CASCADE
                )
            """)
            
            # 学習ログテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    qid INTEGER NOT NULL,
                    selected_choice TEXT NOT NULL,
                    is_correct BOOLEAN NOT NULL,
                    generated_image_path TEXT,
                    session_id TEXT NOT NULL,
                    answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                    FOREIGN KEY (qid) REFERENCES questions (qid) ON DELETE CASCADE
                )
            """)
            
            # 生成画像管理テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qid INTEGER NOT NULL,
                    wrong_choice TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (qid) REFERENCES questions (qid) ON DELETE CASCADE,
                    UNIQUE(qid, wrong_choice)
                )
            """)
            
            # 学習セッション管理テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    mode TEXT CHECK(mode IN ('learning', 'review')) NOT NULL,
                    pos_filter TEXT NOT NULL,
                    cefr_filter TEXT NOT NULL,
                    total_questions INTEGER DEFAULT 10,
                    current_question INTEGER DEFAULT 0,
                    is_completed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            """)
            
            # インデックスの作成（パフォーマンス向上）
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_criteria ON questions(pos, cefr)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_logs_user ON learning_logs(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_logs_session ON learning_logs(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_choices_qid ON choices(qid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generated_images_qid ON generated_images(qid)")
            
        self.logger.info("Database initialized successfully")
    
    # === ユーザー管理 ===
    
    def get_or_create_user(self, username: str) -> int:
        """
        ユーザーを取得または作成
        
        Args:
            username (str): ユーザー名
            
        Returns:
            int: ユーザーID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 既存ユーザーを検索
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            if result:
                return result['user_id']
            
            # 新規ユーザーを作成
            cursor.execute(
                "INSERT INTO users (username) VALUES (?)",
                (username,)
            )
            user_id = cursor.lastrowid
            self.logger.info(f"Created new user: {username} (ID: {user_id})")
            return user_id
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        ユーザーIDからユーザー情報を取得
        
        Args:
            user_id (int): ユーザーID
            
        Returns:
            Optional[Dict]: ユーザー情報
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    # === 問題管理 ===
    
    def save_question(self, question_data: Dict) -> int:
        """
        問題をデータベースに保存
        
        Args:
            question_data (Dict): 問題データ
            
        Returns:
            int: 問題ID (qid)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO questions 
                    (image_id, caption_id, caption, lemma, pos, cefr, answer, blank_question, divided)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    question_data['image_id'],
                    question_data['id'],
                    question_data['caption'],
                    question_data['lemma'],
                    question_data['pos'],
                    question_data['cefr'],
                    question_data['answer'],
                    json.dumps(question_data['blankquestion']),
                    json.dumps(question_data['divided'])
                ))
                
                qid = cursor.lastrowid
                self.logger.info(f"Saved question: {question_data['lemma']} (QID: {qid})")
                return qid
                
            except sqlite3.IntegrityError:
                # 既存の問題が存在する場合、そのqidを返す
                cursor.execute(
                    "SELECT qid FROM questions WHERE lemma = ? AND pos = ? AND cefr = ?",
                    (question_data['lemma'], question_data['pos'], question_data['cefr'])
                )
                result = cursor.fetchone()
                return result['qid'] if result else None
    
    def get_question_by_id(self, qid: int) -> Optional[Dict]:
        """
        問題IDから問題データを取得
        
        Args:
            qid (int): 問題ID
            
        Returns:
            Optional[Dict]: 問題データ
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM questions WHERE qid = ?", (qid,))
            result = cursor.fetchone()
            
            if result:
                question = dict(result)
                # JSON文字列をリストに変換
                question['blankquestion'] = json.loads(question['blank_question'])
                question['divided'] = json.loads(question['divided'])
                del question['blank_question']  # 元のキーを削除
                return question
            
            return None
    
    def get_question_by_criteria(self, pos: str, cefr: str, exclude_qids: List[int] = None) -> Optional[Dict]:
        """
        条件に基づいて問題を取得（ランダム選択）
        
        Args:
            pos (str): 品詞
            cefr (str): CEFR レベル
            exclude_qids (List[int]): 除外する問題IDリスト
            
        Returns:
            Optional[Dict]: 問題データ
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM questions WHERE pos = ? AND cefr = ?"
            params = [pos.lower(), cefr.upper()]
            
            if exclude_qids:
                placeholders = ",".join("?" * len(exclude_qids))
                query += f" AND qid NOT IN ({placeholders})"
                params.extend(exclude_qids)
            
            query += " ORDER BY RANDOM() LIMIT 1"
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result:
                question = dict(result)
                question['blankquestion'] = json.loads(question['blank_question'])
                question['divided'] = json.loads(question['divided'])
                del question['blank_question']
                return question
            
            return None
    
    def question_exists(self, lemma: str, pos: str, cefr: str) -> bool:
        """
        指定された条件の問題が存在するかチェック
        
        Args:
            lemma (str): 見出し語
            pos (str): 品詞
            cefr (str): CEFRレベル
            
        Returns:
            bool: 存在する場合True
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM questions WHERE lemma = ? AND pos = ? AND cefr = ?",
                (lemma, pos.lower(), cefr.upper())
            )
            result = cursor.fetchone()
            return result['count'] > 0
    
    # === 選択肢管理 ===
    
    def save_choices(self, qid: int, choices: List[str], correct_answer: str) -> None:
        """
        選択肢をデータベースに保存
        
        Args:
            qid (int): 問題ID
            choices (List[str]): 選択肢リスト
            correct_answer (str): 正解
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 既存の選択肢を削除
            cursor.execute("DELETE FROM choices WHERE qid = ?", (qid,))
            
            # 新しい選択肢を保存
            for i, choice in enumerate(choices):
                cursor.execute("""
                    INSERT INTO choices (qid, choice_text, is_correct, choice_order)
                    VALUES (?, ?, ?, ?)
                """, (qid, choice, choice == correct_answer, i))
            
            self.logger.info(f"Saved {len(choices)} choices for question {qid}")
    
    def get_choices_by_qid(self, qid: int) -> List[str]:
        """
        問題IDから選択肢を取得
        
        Args:
            qid (int): 問題ID
            
        Returns:
            List[str]: 選択肢リスト
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT choice_text FROM choices WHERE qid = ? ORDER BY choice_order",
                (qid,)
            )
            results = cursor.fetchall()
            return [row['choice_text'] for row in results]
    
    def get_correct_answer(self, qid: int) -> Optional[str]:
        """
        問題IDから正解を取得
        
        Args:
            qid (int): 問題ID
            
        Returns:
            Optional[str]: 正解
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT choice_text FROM choices WHERE qid = ? AND is_correct = TRUE",
                (qid,)
            )
            result = cursor.fetchone()
            return result['choice_text'] if result else None
    
    # === 学習ログ管理 ===
    
    def save_learning_log(self, log_data: Dict) -> None:
        """
        学習ログを保存
        
        Args:
            log_data (Dict): ログデータ
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learning_logs 
                (user_id, qid, selected_choice, is_correct, generated_image_path, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                log_data['user_id'],
                log_data['qid'],
                log_data['selected_choice'],
                log_data['is_correct'],
                log_data.get('generated_image_path'),
                log_data['session_id']
            ))
            
            self.logger.info(f"Saved learning log for user {log_data['user_id']}, question {log_data['qid']}")
    
    def get_user_learning_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        ユーザーの学習履歴を取得
        
        Args:
            user_id (int): ユーザーID
            limit (int): 取得件数制限
            
        Returns:
            List[Dict]: 学習履歴
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ll.*, q.lemma, q.pos, q.cefr, q.caption
                FROM learning_logs ll
                JOIN questions q ON ll.qid = q.qid
                WHERE ll.user_id = ?
                ORDER BY ll.answered_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    def get_review_questions(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        復習用の問題を取得（過去に間違えた問題を優先）- 改善版
        
        Args:
            user_id (int): ユーザーID
            limit (int): 取得件数
            
        Returns:
            List[Dict]: 復習問題リスト（選択肢付き）
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT q.*, ll.is_correct
                FROM questions q
                JOIN learning_logs ll ON q.qid = ll.qid
                WHERE ll.user_id = ?
                ORDER BY ll.is_correct ASC, ll.answered_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = cursor.fetchall()
            questions = []
            for row in results:
                question = dict(row)
                # JSON文字列をリストに変換
                question['blankquestion'] = json.loads(question['blank_question'])
                question['divided'] = json.loads(question['divided'])
                del question['blank_question']
                
                # 修正点: 選択肢も一緒に取得
                cursor.execute(
                    "SELECT choice_text FROM choices WHERE qid = ? ORDER BY choice_order",
                    (question['qid'],)
                )
                choice_results = cursor.fetchall()
                if choice_results:
                    choices = [choice_row['choice_text'] for choice_row in choice_results]
                    question['choices'] = choices
                    question['candidate'] = [c for c in choices if c != question['answer']]
                else:
                    # 選択肢がない場合はログに記録
                    self.logger.warning(f"No choices found for review question QID {question['qid']}")
                    question['choices'] = []
                    question['candidate'] = []
                
                questions.append(question)
            
            return questions
    
    # def get_review_questions(self, user_id: int, limit: int = 10) -> List[Dict]:
    #     """
    #     復習用の問題を取得（過去に間違えた問題を優先）
        
    #     Args:
    #         user_id (int): ユーザーID
    #         limit (int): 取得件数
            
    #     Returns:
    #         List[Dict]: 復習問題リスト
    #     """
    #     with self.get_connection() as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("""
    #             SELECT DISTINCT q.*, ll.is_correct
    #             FROM questions q
    #             JOIN learning_logs ll ON q.qid = ll.qid
    #             WHERE ll.user_id = ?
    #             ORDER BY ll.is_correct ASC, ll.answered_at DESC
    #             LIMIT ?
    #         """, (user_id, limit))
            
    #         results = cursor.fetchall()
    #         questions = []
    #         for row in results:
    #             question = dict(row)
    #             question['blankquestion'] = json.loads(question['blank_question'])
    #             question['divided'] = json.loads(question['divided'])
    #             del question['blank_question']
    #             questions.append(question)
            
    #         return questions
    
    # === 学習セッション管理 ===
    
    def create_learning_session(self, user_id: int, mode: str, pos: str, cefr: str, total_questions: int = 10) -> str:
        """
        学習セッションを作成
        
        Args:
            user_id (int): ユーザーID
            mode (str): 学習モード ('learning' or 'review')
            pos (str): 品詞
            cefr (str): CEFRレベル
            total_questions (int): 問題数
            
        Returns:
            str: セッションID
        """
        session_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learning_sessions 
                (session_id, user_id, mode, pos_filter, cefr_filter, total_questions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, mode, pos.lower(), cefr.upper(), total_questions))
            
            self.logger.info(f"Created learning session: {session_id} for user {user_id}")
            return session_id
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        セッション情報を取得
        
        Args:
            session_id (str): セッションID
            
        Returns:
            Optional[Dict]: セッション情報
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learning_sessions WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def update_session_progress(self, session_id: str) -> None:
        """
        セッションの進捗を更新
        
        Args:
            session_id (str): セッションID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE learning_sessions 
                SET current_question = current_question + 1
                WHERE session_id = ?
            """, (session_id,))
    
    def complete_session(self, session_id: str) -> None:
        """
        セッションを完了状態に設定
        
        Args:
            session_id (str): セッションID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE learning_sessions 
                SET is_completed = TRUE
                WHERE session_id = ?
            """, (session_id,))
            
            self.logger.info(f"Completed learning session: {session_id}")
    
    def get_session_questions_answered(self, session_id: str) -> List[int]:
        """
        セッションで回答済みの問題IDリストを取得
        
        Args:
            session_id (str): セッションID
            
        Returns:
            List[int]: 回答済み問題IDリスト
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT qid FROM learning_logs WHERE session_id = ?",
                (session_id,)
            )
            results = cursor.fetchall()
            return [row['qid'] for row in results]
    
    # === 生成画像管理 ===
    
    def save_generated_image(self, qid: int, wrong_choice: str, image_path: str) -> None:
        """
        生成された画像情報を保存
        
        Args:
            qid (int): 問題ID
            wrong_choice (str): 誤答選択肢
            image_path (str): 画像ファイルパス
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO generated_images (qid, wrong_choice, image_path)
                    VALUES (?, ?, ?)
                """, (qid, wrong_choice, image_path))
                
                self.logger.info(f"Saved generated image for question {qid}, wrong choice: {wrong_choice}")
                
            except sqlite3.IntegrityError:
                # 既存の画像が存在する場合は更新
                cursor.execute("""
                    UPDATE generated_images 
                    SET image_path = ?, created_at = CURRENT_TIMESTAMP
                    WHERE qid = ? AND wrong_choice = ?
                """, (image_path, qid, wrong_choice))
    
    def get_generated_image_path(self, qid: int, wrong_choice: str) -> Optional[str]:
        """
        生成済み画像のパスを取得
        
        Args:
            qid (int): 問題ID
            wrong_choice (str): 誤答選択肢
            
        Returns:
            Optional[str]: 画像ファイルパス
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT image_path FROM generated_images WHERE qid = ? AND wrong_choice = ?",
                (qid, wrong_choice)
            )
            result = cursor.fetchone()
            return result['image_path'] if result else None
    
    # === 統計・分析機能 ===
    
    def get_user_statistics(self, user_id: int) -> Dict:
        """
        ユーザーの学習統計を取得
        
        Args:
            user_id (int): ユーザーID
            
        Returns:
            Dict: 統計情報
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 基本統計
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_questions,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers,
                    COUNT(DISTINCT session_id) as total_sessions
                FROM learning_logs 
                WHERE user_id = ?
            """, (user_id,))
            basic_stats = dict(cursor.fetchone())
            
            # CEFR別統計
            cursor.execute("""
                SELECT 
                    q.cefr,
                    COUNT(*) as attempted,
                    SUM(CASE WHEN ll.is_correct THEN 1 ELSE 0 END) as correct
                FROM learning_logs ll
                JOIN questions q ON ll.qid = q.qid
                WHERE ll.user_id = ?
                GROUP BY q.cefr
            """, (user_id,))
            cefr_stats = {row['cefr']: dict(row) for row in cursor.fetchall()}
            
            # 品詞別統計
            cursor.execute("""
                SELECT 
                    q.pos,
                    COUNT(*) as attempted,
                    SUM(CASE WHEN ll.is_correct THEN 1 ELSE 0 END) as correct
                FROM learning_logs ll
                JOIN questions q ON ll.qid = q.qid
                WHERE ll.user_id = ?
                GROUP BY q.pos
            """, (user_id,))
            pos_stats = {row['pos']: dict(row) for row in cursor.fetchall()}
            
            return {
                'basic': basic_stats,
                'by_cefr': cefr_stats,
                'by_pos': pos_stats
            }
    
    def cleanup_old_sessions(self, days: int = 30) -> None:
        """
        古いセッションデータを削除
        
        Args:
            days (int): 保持日数
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM learning_sessions 
                WHERE created_at < datetime('now', '-{} days')
                AND is_completed = TRUE
            """.format(days))
            
            deleted_count = cursor.rowcount
            self.logger.info(f"Cleaned up {deleted_count} old sessions")


# === 使用例とテスト用コード ===
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # DatabaseManagerのテスト
    db = DatabaseManager("test_vocabulary.db")
    
    # ユーザー作成テスト
    user_id = db.get_or_create_user("test_user")
    print(f"User ID: {user_id}")
    
    # 問題保存テスト
    test_question = {
        'image_id': '123456',
        'id': '789',
        'caption': 'A cat sitting on a table',
        'lemma': 'cat',
        'pos': 'noun',
        'cefr': 'A1',
        'answer': 'cat',
        'blankquestion': ['A', '()', 'sitting', 'on', 'a', 'table'],
        'divided': ['A', 'cat', 'sit', 'on', 'a', 'table']
    }
    
    qid = db.save_question(test_question)
    print(f"Question ID: {qid}")
    
    # 選択肢保存テスト
    choices = ['cat', 'dog', 'bird']
    db.save_choices(qid, choices, 'cat')
    
    # セッション作成テスト
    session_id = db.create_learning_session(user_id, 'learning', 'noun', 'A1')
    print(f"Session ID: {session_id}")
    
    # 学習ログ保存テスト
    log_data = {
        'user_id': user_id,
        'qid': qid,
        'selected_choice': 'cat',
        'is_correct': True,
        'generated_image_path': None,
        'session_id': session_id
    }
    db.save_learning_log(log_data)
    
    # 統計取得テスト
    stats = db.get_user_statistics(user_id)
    print(f"User statistics: {stats}")
    
    print("DatabaseManager test completed successfully!")