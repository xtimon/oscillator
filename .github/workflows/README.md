# GitHub Actions Workflows

Этот каталог содержит GitHub Actions workflows для автоматизации CI/CD процессов.

## Доступные workflows

### 1. `ci.yml` - Continuous Integration
Основной CI пайплайн, который запускается при каждом push и pull request:
- Тестирование на Python 3.9-3.12
- Тестирование на Ubuntu, macOS, Windows
- Проверка импортов всех модулей
- Базовые тесты функциональности
- Линтинг кода (flake8, black)
- Сборка пакета

### 2. `publish.yml` - Публикация в PyPI
Автоматическая публикация пакета в PyPI при создании release:
- Сборка wheel и source distribution
- Публикация в PyPI (требует настройки trusted publisher)

**Настройка:**
1. Настройте Trusted Publisher на [pypi.org](https://pypi.org/manage/account/publishing/)
2. Или создайте API token и добавьте секрет `PYPI_API_TOKEN` в настройках репозитория

### 3. `codeql.yml` - CodeQL Security Analysis
Анализ безопасности кода:
- Автоматический анализ уязвимостей
- Запускается при push и еженедельно

### 4. `docs.yml` - Проверка документации
Проверка качества документации:
- Проверка форматирования README
- Проверка работоспособности примеров кода
- Проверка наличия docstrings

### 5. `dependabot.yml` - Автоматическое обновление зависимостей
Автоматическое обновление зависимостей через Dependabot:
- Еженедельная проверка обновлений pip пакетов
- Автоматическое создание PR для обновлений

## Статус workflows

Вы можете проверить статус всех workflows на вкладке [Actions](../../actions) репозитория.

## Локальный запуск тестов

Для запуска тестов локально:

```bash
# Установить зависимости
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Запустить базовые тесты
python -c "from oscillators import MatterGenesisSimulation; print('OK')"

# Запустить pytest (если есть тесты)
pytest tests/ -v

# С покрытием кода
pytest tests/ -v --cov=oscillators --cov-report=html

# Проверка форматирования
black --check oscillators/

# Линтинг
flake8 oscillators/
```

## Требования для публикации

Перед публикацией убедитесь, что:
1. Версия обновлена в `oscillators/__init__.py`
2. Все тесты проходят
3. Документация актуальна
4. Создан release на GitHub

## Badges

Добавьте эти badges в README.md:

```markdown
[![CI](https://github.com/xtimon/oscillator/actions/workflows/ci.yml/badge.svg)](https://github.com/xtimon/oscillator/actions/workflows/ci.yml)
[![CodeQL](https://github.com/xtimon/oscillator/actions/workflows/codeql.yml/badge.svg)](https://github.com/xtimon/oscillator/actions/workflows/codeql.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
```

