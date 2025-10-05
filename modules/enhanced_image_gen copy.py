# modules/enhanced_image_gen.py

import os
import requests
import hashlib
import time
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Tuple
from database.db_manager import DatabaseManager

class EnhancedImageGenerator:
    """
    データベース連携対応の誤答画像生成クラス
    既存の画像を優先的に再利用し、必要に応じて新規生成する
    """
    
    def __init__(self, db_manager: DatabaseManager, 
                 base_output_dir: str = "generated_images",
                 env_path: str = ".env"):
        """
        EnhancedImageGeneratorを初期化
        
        Args:
            db_manager (DatabaseManager): データベース管理インスタンス
            base_output_dir (str): 画像保存ベースディレクトリ
            env_path (str): 環境変数ファイルのパス
        """
        self.db_manager = db_manager
        self.base_output_dir = base_output_dir
        self.logger = logging.getLogger(__name__)
        
        # 環境変数の読み込み
        load_dotenv(env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("API key is not set. Please define OPENAI_API_KEY in the .env file.")
        
        # OpenAIクライアントの初期化
        self.client = OpenAI(api_key=api_key)
        self.logger.info("OpenAI client initialized successfully")
        
        # ディレクトリの作成
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # 画像生成のパラメータ
        self.image_model = "dall-e-2"
        self.image_size = "256x256"
        self.max_retries = 3
        self.retry_delay = 2  # 秒
        
        # キャッシュとレート制限
        self._generation_cache = {}
        self._last_generation_time = 0
        self.min_generation_interval = 1  # 秒（API制限対応）
        
        # 画像品質設定
        self.image_format = "PNG"
        self.image_quality = 85  # JPEG用（PNGでは無視される）
    
    def get_or_generate_wrong_image(self, qid: int, question_data: Dict, 
                                   selected_answer: str, 
                                   force_regenerate: bool = False) -> str:
        """
        既存の誤答画像を取得するか、新しく生成する（メイン関数）
        
        Args:
            qid (int): 問題ID
            question_data (Dict): 問題データ
            selected_answer (str): 学習者の誤答選択
            force_regenerate (bool): 強制的に再生成するかどうか
            
        Returns:
            str: 画像ファイルパス（相対パス）
        """
        # キャッシュキーの生成
        cache_key = f"image_{qid}_{selected_answer}"
        
        # 強制再生成でない場合、キャッシュとデータベースを確認
        if not force_regenerate:
            # メモリキャッシュから確認
            if cache_key in self._generation_cache:
                self.logger.debug(f"Retrieved image path from cache: {cache_key}")
                return self._generation_cache[cache_key]
            
            # データベースから既存画像を検索
            existing_image_path = self.db_manager.get_generated_image_path(qid, selected_answer)
            if existing_image_path and self._validate_image_file(existing_image_path):
                # キャッシュに保存
                self._generation_cache[cache_key] = existing_image_path
                
                self.logger.info(f"Retrieved existing image for QID {qid}, wrong choice: {selected_answer}")
                return existing_image_path
        
        # 既存画像がない場合、新規生成
        self.logger.info(f"Generating new wrong image for QID {qid}, wrong choice: {selected_answer}")
        new_image_path = self._generate_new_wrong_image(qid, question_data, selected_answer)
        
        if new_image_path:
            # データベースに保存
            self.db_manager.save_generated_image(qid, selected_answer, new_image_path)
            
            # キャッシュに保存
            self._generation_cache[cache_key] = new_image_path
            
            self.logger.info(f"Generated and saved new wrong image: {new_image_path}")
        
        return new_image_path
    
    def _generate_new_wrong_image(self, qid: int, question_data: Dict, 
                                 selected_answer: str) -> Optional[str]:
        """
        新しい誤答画像を生成（既存のgenerate_wrong_image関数を改良）
        
        Args:
            qid (int): 問題ID
            question_data (Dict): 問題データ
            selected_answer (str): 誤答選択
            
        Returns:
            Optional[str]: 生成された画像ファイルパス
        """
        try:
            # プロンプトの生成
            prompt = self._create_enhanced_wrong_prompt(question_data, selected_answer)
            
            # 画像の生成
            image_url = self._generate_image_with_retry(prompt)
            if not image_url:
                return None
            
            # ファイルパスの構築
            image_filename = self._get_enhanced_image_filename(qid, question_data, selected_answer)
            qid_folder = os.path.join(self.base_output_dir, f"qid_{qid}")
            os.makedirs(qid_folder, exist_ok=True)
            
            full_image_path = os.path.join(qid_folder, image_filename)
            relative_path = os.path.join(f"qid_{qid}", image_filename)
            
            # 画像のダウンロードと保存
            success = self._save_image_with_validation(image_url, full_image_path)
            if success:
                return relative_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate wrong image: {e}")
            return None
    
    def _create_enhanced_wrong_prompt(self, question_data: Dict, selected_answer: str) -> str:
        """
        改良された誤答画像生成プロンプトを作成
        
        Args:
            question_data (Dict): 問題データ
            selected_answer (str): 誤答選択
            
        Returns:
            str: 画像生成プロンプト
        """
        base_caption = question_data["caption"]
        lemma = question_data["lemma"]
        pos = question_data.get("pos", "").lower()
        
        # 基本的な置換
        modified_caption = base_caption.replace(lemma, selected_answer)
        
        # 品詞に応じた文脈の調整
        context_enhancers = {
            "noun": f"A clear photo showing a {selected_answer}",
            "verb": f"A photo depicting someone {selected_answer}",
            "adjective": f"A photo showing something {selected_answer}",
            "adverb": f"A photo illustrating an action done {selected_answer}"
        }
        
        # より具体的なプロンプト生成
        if pos in context_enhancers:
            enhanced_prompt = f"{context_enhancers[pos]} in this scene: {modified_caption}"
        else:
            enhanced_prompt = f"A photo depicting the scene: {modified_caption}"
        
        # プロンプトの長さ制限と品質向上
        enhanced_prompt = self._optimize_prompt_for_dalle(enhanced_prompt)
        
        self.logger.debug(f"Generated prompt: {enhanced_prompt}")
        return enhanced_prompt
    
    def _optimize_prompt_for_dalle(self, prompt: str, max_length: int = 400) -> str:
        """
        DALL-E向けにプロンプトを最適化
        
        Args:
            prompt (str): 元のプロンプト
            max_length (int): 最大文字数
            
        Returns:
            str: 最適化されたプロンプト
        """
        # 不適切な単語や表現の除去/置換
        replacements = {
            "inappropriate": "suitable",
            "violent": "peaceful",
            "scary": "friendly",
            "dark": "bright",
            "dangerous": "safe"
        }
        
        optimized = prompt
        for old, new in replacements.items():
            optimized = optimized.replace(old, new)
        
        # 長さの調整
        if len(optimized) > max_length:
            optimized = optimized[:max_length].rsplit(' ', 1)[0] + "."
        
        # 画質向上のためのキーワード追加
        quality_keywords = ["high quality", "clear", "well-lit"]
        if not any(keyword in optimized.lower() for keyword in quality_keywords):
            optimized += ", high quality, clear"
        
        return optimized
    
    def _generate_image_with_retry(self, prompt: str) -> Optional[str]:
        """
        リトライ機能付きの画像生成
        
        Args:
            prompt (str): 画像生成プロンプト
            
        Returns:
            Optional[str]: 生成された画像のURL
        """
        for attempt in range(self.max_retries):
            try:
                # レート制限の考慮
                self._enforce_rate_limit()
                
                response = self.client.images.generate(
                    model=self.image_model,
                    prompt=prompt,
                    size=self.image_size,
                    n=1,
                    response_format="url"
                    # Note: 'quality' parameter removed for DALL-E 2 compatibility
                )
                
                image_url = response.data[0].url
                self.logger.info(f"Successfully generated image on attempt {attempt + 1}")
                return image_url
                
            except Exception as e:
                self.logger.warning(f"Image generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # 指数バックオフ
                else:
                    self.logger.error(f"All {self.max_retries} image generation attempts failed")
        
        return None
    
    def _enforce_rate_limit(self) -> None:
        """
        APIレート制限を強制
        """
        current_time = time.time()
        time_since_last = current_time - self._last_generation_time
        
        if time_since_last < self.min_generation_interval:
            sleep_time = self.min_generation_interval - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self._last_generation_time = time.time()
    
    def _get_enhanced_image_filename(self, qid: int, question_data: Dict, 
                                   selected_answer: str) -> str:
        """
        改良された画像ファイル名を生成
        
        Args:
            qid (int): 問題ID
            question_data (Dict): 問題データ
            selected_answer (str): 誤答選択
            
        Returns:
            str: ファイル名
        """
        # ハッシュベースの安全なファイル名生成
        content_hash = hashlib.md5(
            f"{qid}_{question_data['lemma']}_{selected_answer}".encode()
        ).hexdigest()[:8]
        
        # メタデータを含むファイル名
        filename = (
            f"qid{qid}_{question_data['cefr']}_{question_data['pos']}_"
            f"select_{selected_answer}_{content_hash}.{self.image_format.lower()}"
        )
        
        # ファイル名の安全性確保
        safe_filename = self._sanitize_filename(filename)
        return safe_filename
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        ファイル名を安全な形式に変換
        
        Args:
            filename (str): 元のファイル名
            
        Returns:
            str: 安全なファイル名
        """
        # 不正な文字を除去/置換
        invalid_chars = '<>:"/\\|?*'
        safe_filename = filename
        
        for char in invalid_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # 長さ制限
        max_length = 255
        if len(safe_filename) > max_length:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:max_length - len(ext)] + ext
        
        return safe_filename
    
    def _save_image_with_validation(self, image_url: str, save_path: str) -> bool:
        """
        画像を検証付きで保存
        
        Args:
            image_url (str): 画像URL
            save_path (str): 保存パス
            
        Returns:
            bool: 保存成功時True
        """
        try:
            # 画像のダウンロード
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 画像の検証と変換
            img = Image.open(BytesIO(response.content))
            
            # 画像形式の統一（PNG推奨）
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # 透明度を持つ画像の場合
                img = img.convert("RGBA")
            else:
                # 不透明画像の場合
                img = img.convert("RGB")
            
            # 画像サイズの検証
            expected_size = tuple(map(int, self.image_size.split('x')))
            if img.size != expected_size:
                self.logger.warning(f"Image size mismatch: expected {expected_size}, got {img.size}")
                img = img.resize(expected_size, Image.Resampling.LANCZOS)
            
            # 画像の保存
            if self.image_format.upper() == "PNG":
                img.save(save_path, "PNG", optimize=True)
            else:
                img.save(save_path, "JPEG", quality=self.image_quality, optimize=True)
            
            # ファイルサイズの確認
            file_size = os.path.getsize(save_path)
            if file_size == 0:
                self.logger.error(f"Saved image file is empty: {save_path}")
                return False
            
            self.logger.debug(f"Successfully saved image: {save_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save image from {image_url} to {save_path}: {e}")
            return False
    
    def _validate_image_file(self, image_path: str) -> bool:
        """
        画像ファイルの有効性を検証
        
        Args:
            image_path (str): 画像ファイルパス
            
        Returns:
            bool: 有効な場合True
        """
        full_path = os.path.join(self.base_output_dir, image_path)
        
        try:
            # ファイルの存在確認
            if not os.path.exists(full_path):
                return False
            
            # ファイルサイズの確認
            if os.path.getsize(full_path) == 0:
                return False
            
            # 画像として開けるか確認
            with Image.open(full_path) as img:
                img.verify()  # 画像の整合性を検証
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Image validation failed for {image_path}: {e}")
            return False
    
    def generate_wrong_image(self, question_data: Dict, selected_answer: str) -> Dict:
        """
        既存のgenerate_wrong_image関数との互換性維持
        
        Args:
            question_data (Dict): 問題データ
            selected_answer (str): 学習者の誤答選択
            
        Returns:
            Dict: 更新されたメタデータ
        """
        qid = question_data.get("qid")
        if qid is None:
            raise ValueError("Missing 'qid' in question_data")
        
        # 新しいメソッドを使用
        image_path = self.get_or_generate_wrong_image(qid, question_data, selected_answer)
        
        # 既存の形式でメタデータを返す
        metadata = question_data.copy()
        metadata["select_ans"] = selected_answer
        metadata["learnerans1"] = selected_answer
        metadata["genimage1"] = image_path
        
        return metadata
    
    def get_image_generation_stats(self) -> Dict:
        """
        画像生成の統計情報を取得
        
        Returns:
            Dict: 統計情報
        """
        try:
            # ディレクトリ内の画像ファイル数をカウント
            total_images = 0
            total_size = 0
            qid_folders = []
            
            for item in os.listdir(self.base_output_dir):
                if item.startswith("qid_"):
                    qid_folders.append(item)
                    folder_path = os.path.join(self.base_output_dir, item)
                    if os.path.isdir(folder_path):
                        for filename in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, filename)
                            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                total_images += 1
                                total_size += os.path.getsize(file_path)
            
            return {
                'total_generated_images': total_images,
                'total_storage_size_mb': round(total_size / (1024 * 1024), 2),
                'unique_qid_folders': len(qid_folders),
                'average_file_size_kb': round((total_size / total_images) / 1024, 2) if total_images > 0 else 0,
                'cache_entries': len(self._generation_cache)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get image generation stats: {e}")
            return {'error': str(e)}
    
    def cleanup_invalid_images(self) -> Dict[str, int]:
        """
        無効な画像ファイルをクリーンアップ
        
        Returns:
            Dict[str, int]: クリーンアップ結果
        """
        cleaned_count = 0
        error_count = 0
        
        try:
            for item in os.listdir(self.base_output_dir):
                if item.startswith("qid_"):
                    folder_path = os.path.join(self.base_output_dir, item)
                    if os.path.isdir(folder_path):
                        for filename in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, filename)
                            relative_path = os.path.join(item, filename)
                            
                            if not self._validate_image_file(relative_path):
                                try:
                                    os.remove(file_path)
                                    cleaned_count += 1
                                    self.logger.info(f"Removed invalid image: {relative_path}")
                                except Exception as e:
                                    error_count += 1
                                    self.logger.error(f"Failed to remove invalid image {relative_path}: {e}")
        
        except Exception as e:
            self.logger.error(f"Cleanup process failed: {e}")
            error_count += 1
        
        # キャッシュもクリア
        self.clear_cache()
        
        return {
            'cleaned_files': cleaned_count,
            'errors': error_count
        }
    
    def clear_cache(self) -> None:
        """
        画像生成キャッシュをクリア
        """
        self._generation_cache.clear()
        self.logger.info("Image generation cache cleared")
    
    def get_image_by_criteria(self, qid: int, wrong_choice: str) -> Optional[str]:
        """
        特定の条件で画像パスを取得
        
        Args:
            qid (int): 問題ID
            wrong_choice (str): 誤答選択肢
            
        Returns:
            Optional[str]: 画像パス
        """
        return self.db_manager.get_generated_image_path(qid, wrong_choice)
    
    def regenerate_image(self, qid: int, question_data: Dict, selected_answer: str) -> str:
        """
        特定の画像を強制的に再生成
        
        Args:
            qid (int): 問題ID
            question_data (Dict): 問題データ
            selected_answer (str): 誤答選択
            
        Returns:
            str: 新しい画像パス
        """
        return self.get_or_generate_wrong_image(qid, question_data, selected_answer, force_regenerate=True)
    
    def batch_generate_images(self, generation_requests: List[Tuple[int, Dict, str]]) -> List[Tuple[int, str, Optional[str]]]:
        """
        バッチでの画像生成
        
        Args:
            generation_requests (List[Tuple[int, Dict, str]]): (qid, question_data, selected_answer)のリスト
            
        Returns:
            List[Tuple[int, str, Optional[str]]]: (qid, selected_answer, image_path)のリスト
        """
        results = []
        
        for qid, question_data, selected_answer in generation_requests:
            try:
                image_path = self.get_or_generate_wrong_image(qid, question_data, selected_answer)
                results.append((qid, selected_answer, image_path))
                
                # バッチ処理での適切な間隔を保つ
                time.sleep(self.min_generation_interval)
                
            except Exception as e:
                self.logger.error(f"Batch generation failed for QID {qid}, choice {selected_answer}: {e}")
                results.append((qid, selected_answer, None))
        
        return results


# === 使用例とテスト用コード ===
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # テスト用データベース
    from database.db_manager import DatabaseManager
    
    # DatabaseManagerとEnhancedImageGeneratorの初期化
    db_manager = DatabaseManager("test_enhanced_images.db")
    image_gen = EnhancedImageGenerator(db_manager)
    
    print("=== EnhancedImageGenerator Test ===")
    
    # 1. システム統計の確認
    print("\n1. Image generation statistics:")
    stats = image_gen.get_image_generation_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 2. テスト問題データの作成
    test_question = {
        "qid": 1,
        "image_id": "123456",
        "caption": "A cat sitting on a table",
        "lemma": "cat",
        "pos": "noun",
        "cefr": "A1",
        "answer": "cat"
    }
    
    # 3. 画像生成テスト（実際のAPI呼び出しは避ける）
    print("\n2. Image generation test (simulation):")
    print(f"  Test question: {test_question['lemma']} ({test_question['pos']} {test_question['cefr']})")
    print(f"  Caption: {test_question['caption']}")
    
    # プロンプト生成テスト
    test_wrong_answers = ["dog", "bird", "table"]
    for wrong_answer in test_wrong_answers:
        prompt = image_gen._create_enhanced_wrong_prompt(test_question, wrong_answer)
        print(f"  Wrong answer '{wrong_answer}' → Prompt: {prompt}")
    
    # ファイル名生成テスト
    for wrong_answer in test_wrong_answers:
        filename = image_gen._get_enhanced_image_filename(1, test_question, wrong_answer)
        print(f"  Wrong answer '{wrong_answer}' → Filename: {filename}")
    
    # 4. データベース連携テスト（模擬）
    print("\n3. Database integration test (simulation):")
    
    # 既存画像の確認
    existing_path = image_gen.get_image_by_criteria(1, "dog")
    print(f"  Existing image for QID 1, wrong choice 'dog': {existing_path}")
    
    # 5. ファイル検証テスト
    print("\n4. File validation test:")
    
    # 存在しないファイルの検証
    is_valid = image_gen._validate_image_file("nonexistent/path.png")
    print(f"  Validation of non-existent file: {is_valid}")
    
    # 6. キャッシュテスト
    print("\n5. Cache test:")
    print(f"  Current cache entries: {len(image_gen._generation_cache)}")
    
    # キャッシュのクリア
    image_gen.clear_cache()
    print(f"  Cache entries after clear: {len(image_gen._generation_cache)}")
    
    # 7. バッチ生成テスト（シミュレーション）
    print("\n6. Batch generation test (simulation):")
    batch_requests = [
        (1, test_question, "dog"),
        (1, test_question, "bird"),
        (1, test_question, "table")
    ]
    
    print(f"  Batch requests: {len(batch_requests)} items")
    print("  Note: Actual generation requires valid OpenAI API key and will be skipped in test")
    
    print("\n=== Test completed ===")
    print("Note: This test simulates image generation without actual API calls.")
    print("To test actual image generation, ensure OPENAI_API_KEY is set in .env file.")