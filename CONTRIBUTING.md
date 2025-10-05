# Contributing to L-VEIGe

Thanks for your interest in contributing! We welcome bug reports, feature requests, and pull requests.

## Ways to contribute
- Report bugs or request features via **Issues**
- Improve documentation or examples
- Submit pull requests with fixes or enhancements

## Development setup
```bash
git clone <YOUR_FORK_URL>
cd L-VEIGe_Original
python3.8 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock-py38.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # or follow README to create .env
