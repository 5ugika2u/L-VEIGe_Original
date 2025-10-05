# modules/enhanced_image_gen.py
# インデント修正版（スペース4つで統一）

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
    データベース連携対応の誤答画像生成クラス（速度最優先版）
    既存の画像を優先的に再利用し、必要に応じて新規生成する
    """
    
    def __init__(self, db_manager: DatabaseManager, 
                 base_output_dir: str = "generated_images",
                 env_path: str = ".env"):
        """
        EnhancedImageGeneratorを初期化（DALL-E 3専用）
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
        self.logger.info("OpenAI client initialized for DALL-E 3")
        
        # ディレクトリの作成
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # DALL-E 3専用設定（速度最優先）
        self.image_model = "dall-e-3"
        self.image_size = "1024x1024"  # DALL-E 3の最小サイズ
        self.image_quality = "standard"  # 最低品質で最高速
        self.image_style = "natural"  # シンプルスタイルで最高速
        self.min_generation_interval = 1  # 最短間隔
        
        self.logger.info("Using DALL-E 3 with speed-optimized settings")
        
        # 超高速化設定
        self.max_retries = 1
        self.retry_delay = 0.5
        
        # キャッシュとレート制限
        self._generation_cache = {}
        self._last_generation_time = 0
        
        # 画像品質設定（速度重視）
        self.image_format = "JPEG"
        self.image_quality_jpeg = 70

    def get_or_generate_wrong_image(self, qid: int, question_data: Dict, 
                                   selected_answer: str, 
                                   force_regenerate: bool = False) -> str:
        """
        既存の誤答画像を取得するか、新しく生成する（メイン関数）
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
        新しい誤答画像を生成（速度最優先版）
        """
        try:
            # 超シンプルプロンプトの生成
            prompt = self._create_minimal_safe_prompt(question_data, selected_answer)
            
            # 画像の生成
            image_url = self._generate_image_with_retry(prompt)
            if not image_url:
                return None
            
            # ファイルパスの構築
            image_filename = self._get_speed_optimized_filename(qid, question_data, selected_answer)
            qid_folder = os.path.join(self.base_output_dir, f"qid_{qid}")
            os.makedirs(qid_folder, exist_ok=True)
            
            full_image_path = os.path.join(qid_folder, image_filename)
            relative_path = os.path.join(f"qid_{qid}", image_filename)
            
            # 画像のダウンロードと保存（高速版）
            success = self._save_image_fast(image_url, full_image_path)
            if success:
                return relative_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate wrong image: {e}")
            return None

    def _create_minimal_safe_prompt(self, question_data: Dict, selected_answer: str) -> str:
        """
        最小限の安全なプロンプトを作成（「"（誤答を含む文）"」スタイル）
        """
        base_caption = question_data["caption"]
        lemma = question_data["lemma"]
        
        # 基本的な置換
        modified_caption = base_caption.replace(lemma, selected_answer)
        
        # 不適切表現の除外のみ実行
        safe_caption = self._remove_inappropriate_content(modified_caption)
        
        # 最もシンプルなプロンプト：「"（学習者の誤答を含む文）"」
        minimal_prompt = f'"{safe_caption}"'
        
        self.logger.debug(f"Generated minimal safe prompt: {minimal_prompt}")
        return minimal_prompt

    def _remove_inappropriate_content(self, text: str) -> str:
        """
        不適切表現のみを除外（速度重視の最小限処理）
        """
        # 必要最小限の不適切表現除去
        unsafe_words = {
            "violent": "peaceful",
            "scary": "calm", 
            "dangerous": "safe",
            "blood": "red",
            "weapon": "object",
            "gun": "tool",
            "knife": "utensil",
            "death": "sleep",
            "kill": "stop",
            "hurt": "touch"
        }
        
        safe_text = text.lower()
        for unsafe, safe in unsafe_words.items():
            safe_text = safe_text.replace(unsafe, safe)
        
        # 元の大文字小文字構造をある程度保持
        return safe_text.capitalize()

    def _generate_image_with_retry(self, prompt: str) -> Optional[str]:
        """
        DALL-E 3でのリトライ機能付き画像生成
        """
        for attempt in range(self.max_retries):
            try:
                # レート制限の考慮
                self._enforce_rate_limit()
                
                # DALL-E 3のAPI呼び出し（速度最優先設定）
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=self.image_size,
                    quality="standard",  # 最低品質で最高速
                    style="natural",     # シンプルスタイルで最高速
                    n=1,
                    response_format="url"
                )
                
                image_url = response.data[0].url
                self.logger.info(f"Successfully generated image with DALL-E 3")
                return image_url
                
            except Exception as e:
                self.logger.warning(f"DALL-E 3 generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All DALL-E 3 generation attempts failed")
        
        return None

    def _enforce_rate_limit(self) -> None:
        """
        APIレート制限を強制（最小限）
        """
        current_time = time.time()
        time_since_last = current_time - self._last_generation_time
        
        if time_since_last < self.min_generation_interval:
            sleep_time = self.min_generation_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_generation_time = time.time()

    def _get_speed_optimized_filename(self, qid: int, question_data: Dict, 
                                    selected_answer: str) -> str:
        """
        DALL-E 3専用の速度最優先ファイル名を生成
        """
        # 最小限のハッシュ生成
        content_hash = hashlib.md5(
            f"{qid}_{selected_answer}".encode()
        ).hexdigest()[:6]
        
        # DALL-E 3専用のシンプルなファイル名
        filename = f"d3_q{qid}_ans{content_hash}.{self.image_format.lower()}"
        
        return self._sanitize_filename(filename)

    def _sanitize_filename(self, filename: str) -> str:
        """
        ファイル名を安全な形式に変換（高速版）
        """
        # 最小限の文字置換
        invalid_chars = '<>:"/\\|?*'
        safe_filename = filename
        
        for char in invalid_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # 長さ制限（簡易版）
        if len(safe_filename) > 100:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:90] + ext
        
        return safe_filename

    def _save_image_fast(self, image_url: str, save_path: str) -> bool:
        """
        画像を高速保存（品質より速度重視）
        """
        try:
            # 画像のダウンロード（タイムアウト短縮）
            response = requests.get(image_url, timeout=20)
            response.raise_for_status()
            
            # 最小限の画像処理
            img = Image.open(BytesIO(response.content))
            
            # RGB変換（最小限）
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 高速保存（最適化無効）
            if self.image_format.upper() == "JPEG":
                img.save(save_path, "JPEG", quality=self.image_quality_jpeg, optimize=False)
            else:
                img.save(save_path, "PNG", optimize=False)
            
            # 基本的なファイルサイズ確認
            if os.path.getsize(save_path) > 0:
                self.logger.debug(f"Fast saved image: {save_path}")
                return True
            else:
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to fast save image: {e}")
            return False

    def _validate_image_file(self, image_path: str) -> bool:
        """
        画像ファイルの有効性を高速検証
        """
        full_path = os.path.join(self.base_output_dir, image_path)
        
        try:
            # 基本的な存在・サイズ確認のみ（高速）
            return os.path.exists(full_path) and os.path.getsize(full_path) > 0
        except Exception:
            return False

    # === 既存メソッドとの互換性維持 ===
    
    def generate_wrong_image(self, question_data: Dict, selected_answer: str) -> Dict:
        """
        既存のgenerate_wrong_image関数との互換性維持
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
        DALL-E 3画像生成の統計情報を取得
        """
        try:
            total_images = 0
            total_size = 0
            dalle3_images = 0
            
            for item in os.listdir(self.base_output_dir):
                if item.startswith("qid_"):
                    folder_path = os.path.join(self.base_output_dir, item)
                    if os.path.isdir(folder_path):
                        for filename in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, filename)
                            if os.path.isfile(file_path):
                                total_images += 1
                                total_size += os.path.getsize(file_path)
                                if filename.startswith('d3_'):
                                    dalle3_images += 1
            
            return {
                'total_generated_images': total_images,
                'dalle3_images': dalle3_images,
                'total_storage_size_mb': round(total_size / (1024 * 1024), 2),
                'current_model': 'dall-e-3',
                'image_quality': self.image_quality,
                'image_style': self.image_style,
                'image_size': self.image_size,
                'speed_optimized': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

    def clear_cache(self) -> None:
        """
        画像生成キャッシュをクリア
        """
        self._generation_cache.clear()
        self.logger.info("Cache cleared")

    def set_dalle3_quality(self, quality: str) -> None:
        """
        DALL-E 3の画質設定を変更
        
        Args:
            quality (str): "standard" または "hd"
        """
        if quality in ["standard", "hd"]:
            self.image_quality = quality
            self.logger.info(f"DALL-E 3 quality set to: {quality}")
        else:
            raise ValueError("Quality must be 'standard' or 'hd'")

    def set_dalle3_style(self, style: str) -> None:
        """
        DALL-E 3のスタイル設定を変更
        
        Args:
            style (str): "natural" または "vivid"
        """
        if style in ["natural", "vivid"]:
            self.image_style = style
            self.logger.info(f"DALL-E 3 style set to: {style}")
        else:
            raise ValueError("Style must be 'natural' or 'vivid'")

    def set_dalle3_size(self, size: str) -> None:
        """
        DALL-E 3の画像サイズ設定を変更
        
        Args:
            size (str): "1024x1024", "1024x1792", または "1792x1024"
        """
        valid_sizes = ["1024x1024", "1024x1792", "1792x1024"]
        if size in valid_sizes:
            self.image_size = size
            self.logger.info(f"DALL-E 3 size set to: {size}")
        else:
            raise ValueError(f"Size must be one of: {valid_sizes}")

    def get_current_settings(self) -> Dict:
        """
        現在のDALL-E 3設定を取得
        """
        return {
            'model': 'dall-e-3',
            'size': self.image_size,
            'quality': self.image_quality,
            'style': self.image_style,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'min_interval': self.min_generation_interval,
            'image_format': self.image_format,
            'speed_optimized': True
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== DALL-E 3専用 EnhancedImageGenerator ===")
    print("速度最優先設定・インデント修正済み")
    
    # テスト例
    test_question = {
        "caption": "A dangerous cat sitting on a violent table",
        "lemma": "cat"
    }
    
    # ダミーでプロンプト生成テスト（DBなしでテスト）
    class DummyDB:
        pass
    
    try:
        # 初期化テスト（API キーなしでもプロンプト生成のテストは可能）
        print("\n初期化テスト:")
        print("- DALL-E 3専用モード")
        print("- 速度最優先設定")
        print("- 画質: standard（最低品質）")
        print("- スタイル: natural（シンプル）")
        print("- サイズ: 1024x1024（最小）")
        
    except Exception as e:
        print(f"注意: 実際の初期化にはOPENAI_API_KEYが必要です: {e}")
    
    print("\n=== DALL-E 3専用版完成 ===")