# Django Project: Отчеты

Этот проект представляет собой Django приложение для генерации отчетов.

## Требования

- Python 3.7+
- Django 3.x+
- pip (Python package installer)

## Установка

Следуйте инструкциям ниже, чтобы развернуть проект на новой машине.

### Клонирование репозитория

Сначала клонируйте репозиторий на вашу локальную машину:

```bash
git clone https://github.com/ваш-пользователь/ваш-репозиторий.git
cd ваш-репозиторий
```
### Создание виртуального окружения
Рекомендуется использовать виртуальное окружение для изоляции зависимостей проекта. Установите и активируйте виртуальное окружение:
```bash
python3 -m venv venv
source venv/bin/activate  # для Windows используйте `venv\Scripts\activate`
```
Установите все необходимые библиотеки, перечисленные в файле requirements.txt:
```bash
pip install -r requirements.txt
```

### Настройка БД:
Откройте файл settings.py и внесите изменения в конфигурацию базы данных в соответствии с вашей используемой базой данных. Например:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',  # или другой используемый вами движок
        'NAME': 'ваше_имя_базы_данных',
        'USER': 'ваш_пользователь',
        'PASSWORD': 'ваш_пароль',
        'HOST': 'ваш_хост',
        'PORT': 'ваш_порт',
    }
}
```

### Применение миграций
Примените миграции для создания необходимых таблиц в базе данных:
```bash
python manage.py makemigrations
python manage.py migrate
```

### Создание суперпользователя
Создайте суперпользователя для доступа к административной панели Django:
```bash
python manage.py createsuperuser
```
### Запуск сервера
Запустите локальный сервер разработки:
```bash
python manage.py runserver
```
Теперь ваш проект должен быть доступен по адресу http://127.0.0.1:8000/.