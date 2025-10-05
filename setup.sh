#!/bin/bash
# setup.sh - Automated setup script for the Vocabulary Learning System

set -e  # Stop the script on any error

echo "🎓 Vocabulary Learning System Setup"
echo "==================================="

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Found Python $python_version"
else
    echo "❌ Python 3.8 or higher is required (current: $python_version)"
    exit 1
fi

# Create virtual environment
echo ""
echo "2. Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install packages
echo ""
echo "4. Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Package installation complete"

# Download SpaCy English model
echo ""
echo "5. Downloading SpaCy English model..."
python -m spacy download en_core_web_sm
echo "✓ SpaCy English model downloaded"

# Create directory structure
echo ""
echo "6. Creating directory structure..."
mkdir -p static/images
mkdir -p data
mkdir -p templates
mkdir -p database
mkdir -p modules
mkdir -p generated_images
mkdir -p tests
echo "✓ Directory structure created"

# Create __init__.py files
echo ""
echo "7. Creating Python module files..."
touch database/__init__.py
touch modules/__init__.py
echo "✓ __init__.py files created"

# Create .env file
echo ""
echo "8. Checking environment file..."
if [ ! -f ".env" ]; then
    echo "FLASK_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > .env
    echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
    echo "DATABASE_PATH=vocabulary_learning.db" >> .env
    echo "FLASK_DEBUG=True" >> .env
    echo "FLASK_ENV=development" >> .env
    echo "QUESTIONS_PER_SESSION=10" >> .env
    echo "✓ Created .env file"
    echo "⚠️  Please set OPENAI_API_KEY in your .env file"
else
    echo "✓ .env file already exists"
fi

# Create .gitignore
echo ""
echo "9. Creating .gitignore..."
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << EOF
# Environment variables
.env
*.env

# Databases
*.db
*.sqlite
*.sqlite3

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Generated files
generated_images/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Tests
.pytest_cache/
.coverage
htmlcov/

# Backups
*.backup
*.bak
EOF
    echo "✓ .gitignore created"
else
    echo "✓ .gitignore already exists"
fi

# Create a placeholder image
echo ""
echo "10. Creating placeholder image..."
python3 << 'EOF'
from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder():
    width, height = 400, 300
    img = Image.new('RGB', (width, height), color='#f0f0f0')
    draw = ImageDraw.Draw(img)

    draw.rectangle([10, 10, width-10, height-10], outline='#cccccc', width=2)

    text = "Image Not Found"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = 200, 40

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill='#999999', font=font, align='center')

    os.makedirs('static', exist_ok=True)
    img.save('static/placeholder.jpg', 'JPEG', quality=85)

create_placeholder()
print("✓ Placeholder image created")
EOF

# Run system test
echo ""
echo "11. Running system tests..."
if python3 test_system.py > /dev/null 2>&1; then
    echo "✓ System tests passed"
else
    echo "⚠️  Issues detected during system tests"
    echo "   See details: python test_system.py"
fi

# Done
echo ""
echo "🎉 Setup complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Set OPENAI_API_KEY in .env"
echo "   nano .env"
echo ""
echo "2. Place COCO image files into static/images/"
echo "   e.g., static/images/000000017379.jpg"
echo ""
echo "3. Run system tests"
echo "   python test_system.py"
echo ""
echo "4. Start the application"
echo "   python app.py"
echo ""
echo "5. Open your browser"
echo "   http://127.0.0.1:5000"
echo ""
echo "Troubleshooting: see README.md"
