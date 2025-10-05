# create_placeholder.py - プレースホルダー画像作成スクリプト

from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_image():
    """
    プレースホルダー画像を作成
    """
    # 画像サイズ
    width, height = 400, 300
    
    # 画像を作成
    img = Image.new('RGB', (width, height), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # テキストを描画
    try:
        # システムフォントを使用
        font = ImageFont.load_default()
    except:
        font = None
    
    # 背景色とボーダー
    draw.rectangle([10, 10, width-10, height-10], outline='#cccccc', width=2)
    
    # テキスト
    text = "画像が見つかりません\nImage Not Found"
    
    # テキストサイズを計算
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = 200, 40
    
    # テキストを中央に配置
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.multiline_text((x, y), text, fill='#999999', font=font, align='center')
    
    # ディレクトリを作成
    os.makedirs('static', exist_ok=True)
    
    # 画像を保存
    img.save('static/placeholder.jpg', 'JPEG', quality=85)
    print("プレースホルダー画像を作成しました: static/placeholder.jpg")

if __name__ == "__main__":
    create_placeholder_image()