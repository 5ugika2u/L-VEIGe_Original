# test_system.py - システム総合テストスクリプト

import os
import sys
import json
import pandas as pd
from pathlib import Path

def test_file_structure():
    """
    ファイル構造をテスト
    """
    print("=== 1. ファイル構造テスト ===")
    
    required_files = [
        'app.py',
        '.env',
        'data/captions_val2017_sample10.json',
        'data/coco_cefr_vocab.csv',
        'database/db_manager.py',
        'modules/enhanced_question_gen.py',
        'modules/enhanced_candidate_gen.py',
        'modules/enhanced_image_gen.py',
        'modules/result_processor.py',
        'templates/base.html',
        'templates/login.html',
        'templates/question.html',
        'templates/result.html',
        'static/placeholder.jpg'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} (見つかりません)")
    
    # 画像ファイルの確認
    print("\n画像ファイルの確認:")
    image_dir = "static/images"
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        print(f"✓ 画像ディレクトリ存在: {len(image_files)}個のJPGファイル")
        for img in image_files[:5]:  # 最初の5個だけ表示
            print(f"  - {img}")
        if len(image_files) > 5:
            print(f"  ... 他{len(image_files)-5}個")
    else:
        print(f"✗ 画像ディレクトリが見つかりません: {image_dir}")
        missing_files.append(image_dir)
    
    print(f"\n結果: {len(existing_files)}個のファイルが存在, {len(missing_files)}個が不足")
    return len(missing_files) == 0

def test_data_integrity():
    """
    データファイルの整合性をテスト
    """
    print("\n=== 2. データ整合性テスト ===")
    
    try:
        # キャプションデータの確認
        with open('data/captions_val2017_sample10.json', 'r') as f:
            caption_data = json.load(f)
        
        print(f"✓ キャプションデータ読み込み成功")
        print(f"  - 画像数: {len(caption_data.get('images', []))}")
        print(f"  - キャプション数: {len(caption_data.get('annotations', []))}")
        
        # 語彙データの確認
        vocab_df = pd.read_csv('data/coco_cefr_vocab.csv')
        print(f"✓ 語彙データ読み込み成功")
        print(f"  - 総語彙数: {len(vocab_df)}")
        print(f"  - 品詞: {vocab_df['POS'].value_counts().to_dict()}")
        print(f"  - CEFR: {vocab_df['CEFR'].value_counts().to_dict()}")
        
        # データの整合性確認
        caption_ids = set(ann['id'] for ann in caption_data['annotations'])
        vocab_caption_ids = set(vocab_df['CaptionID'].unique())
        vocab_caption_ids.discard(0)  # プレースホルダーを除外
        
        matching_ids = caption_ids.intersection(vocab_caption_ids)
        print(f"✓ データ整合性: {len(matching_ids)}個のキャプションIDが一致")
        
        if len(matching_ids) > 0:
            print("  一致するキャプションID例:", list(matching_ids)[:3])
        
        return True
        
    except Exception as e:
        print(f"✗ データ整合性テスト失敗: {e}")
        return False

def test_environment():
    """
    環境変数をテスト
    """
    print("\n=== 3. 環境変数テスト ===")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_env = ['FLASK_SECRET_KEY', 'OPENAI_API_KEY']
    missing_env = []
    
    for env_var in required_env:
        value = os.getenv(env_var)
        if value:
            if env_var == 'FLASK_SECRET_KEY':
                print(f"✓ {env_var}: 設定済み (長さ: {len(value)})")
            elif env_var == 'OPENAI_API_KEY':
                masked = value[:8] + '*' * (len(value) - 12) + value[-4:]
                print(f"✓ {env_var}: {masked}")
        else:
            print(f"✗ {env_var}: 未設定")
            missing_env.append(env_var)
    
    return len(missing_env) == 0

def test_imports():
    """
    モジュールインポートをテスト
    """
    print("\n=== 4. モジュールインポートテスト ===")
    
    modules_to_test = [
        ('database.db_manager', 'DatabaseManager'),
        ('modules.enhanced_question_gen', 'EnhancedQuestionGenerator'),
        ('modules.enhanced_candidate_gen', 'EnhancedCandidateGenerator'),
        ('modules.enhanced_image_gen', 'EnhancedImageGenerator'),
        ('modules.result_processor', 'ResultProcessor')
    ]
    
    import_errors = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name}: {e}")
            import_errors.append(f"{module_name}.{class_name}")
    
    return len(import_errors) == 0

def test_database_creation():
    """
    データベース作成をテスト
    """
    print("\n=== 5. データベース作成テスト ===")
    
    try:
        # テスト用データベースファイル
        test_db_path = "test_vocabulary.db"
        
        # 既存のテストDBを削除
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        from database.db_manager import DatabaseManager
        
        # データベース初期化
        db_manager = DatabaseManager(test_db_path)
        print("✓ データベース初期化成功")
        
        # テストユーザー作成
        user_id = db_manager.get_or_create_user("test_user")
        print(f"✓ ユーザー作成成功: ID {user_id}")
        
        # テストセッション作成
        session_id = db_manager.create_learning_session(user_id, "learning", "noun", "A1")
        print(f"✓ セッション作成成功: {session_id}")
        
        # クリーンアップ
        os.remove(test_db_path)
        print("✓ テストデータベース削除完了")
        
        return True
        
    except Exception as e:
        print(f"✗ データベーステスト失敗: {e}")
        return False

def test_question_generation():
    """
    問題生成をテスト
    """
    print("\n=== 6. 問題生成テスト ===")
    
    try:
        from database.db_manager import DatabaseManager
        from modules.enhanced_question_gen import EnhancedQuestionGenerator
        
        # テスト用データベース
        test_db_path = "test_question_gen.db"
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        db_manager = DatabaseManager(test_db_path)
        question_gen = EnhancedQuestionGenerator(db_manager)
        
        print("✓ 問題生成モジュール初期化成功")
        
        # 利用可能な条件を取得
        criteria = question_gen.get_available_criteria()
        print(f"✓ 利用可能な条件: {len(criteria)}種類の品詞")
        
        # 問題生成テスト
        for pos in list(criteria.keys())[:2]:  # 最初の2つの品詞でテスト
            for cefr in criteria[pos][:2]:  # 最初の2つのCEFRレベルでテスト
                try:
                    question = question_gen.get_or_generate_question(pos, cefr)
                    if question:
                        print(f"✓ 問題生成成功: {pos} {cefr} - {question.get('lemma', 'N/A')}")
                    else:
                        print(f"- 問題生成なし: {pos} {cefr}")
                except Exception as e:
                    print(f"✗ 問題生成エラー {pos} {cefr}: {e}")
        
        # クリーンアップ
        os.remove(test_db_path)
        return True
        
    except Exception as e:
        print(f"✗ 問題生成テスト失敗: {e}")
        return False

def run_all_tests():
    """
    全てのテストを実行
    """
    print("🎯 語彙学習システム総合テスト開始\n")
    
    tests = [
        ("ファイル構造", test_file_structure),
        ("データ整合性", test_data_integrity),
        ("環境変数", test_environment),
        ("モジュールインポート", test_imports),
        ("データベース作成", test_database_creation),
        ("問題生成", test_question_generation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name}テストで予期しないエラー: {e}")
            results[test_name] = False
    
    # 結果サマリー
    print("\n" + "="*50)
    print("🏁 テスト結果サマリー")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 合格" if result else "❌ 不合格"
        print(f"{test_name}: {status}")
    
    print(f"\n総合結果: {passed}/{total} テスト合格")
    
    if passed == total:
        print("🎉 全てのテストに合格！Flaskアプリケーションを起動できます。")
        print("\n次のコマンドでアプリケーションを起動:")
        print("python app.py")
    else:
        print("⚠️  いくつかのテストが失敗しました。上記のエラーを修正してください。")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()