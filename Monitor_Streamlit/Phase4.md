# Book Purchase Intention Prediction

## Phase 4: Testing and CI/CD Automation

- Install `pytest` packages

  ```bash
  python3 -m pip install --upgrade pip
  python3 -m pip install pytest pytest-asyncio
  ```

## 4.1. Comprehensive Testing

There are individual test file in the four folders: `FastAPI_Backend`, `Model_Management`, `Monitor_Streamlit`, `Streamlit_Frontend`.

**Unit Tests**

`test_backend.py` in the `FastAPI_Backend`: 

1. `test_ensure_table_existing` tests if `ensure_table` function in the program can load existed table well.
2. `test_ensure_table_creates_if_missing` tests if `ensure_table` function creates new table well.

`test_manage.py` in the `Model_Management`:

3. `test_log_artifact` tests if `log_artifact` function in the program can create wandb artifact for data and model well.

`test_dashboard.py` in the `Monitor_Streamlit`:

4. `test_ensure_table_existing` tests if `ensure_table` function in the program can load existed table well.
5. `test_ensure_table_creates_if_missing` tests if `ensure_table` function creates new table well.

`test_frontend.py`  in the `Streamlit_Frontend`:

6. `test_backend_predict` tests if `backend_predict` function retrieves the prediction from FastAPI backend well.

**Integration Tests**

`test_backend.py` in the `FastAPI_Backend`: 

1. `test_predict2` tests if program runs endpoints `predict` well.
2. `test_predict3` tests if program captures wrong input type and missing input value well.

`test_manage.py` in the `Model_Management`:

3. `test_main` tests if program runs the entire `main()` well.

`test_dashboard.py` in the `Monitor_Streamlit`:

4. `test_main` tests if program runs the entire `main()` well.

# 4.2. CI/CD Pipeline

We created `.github/workflows/ci.yml` under `./4705_Book_Purchase_Intention_Prediction`

In the `ci.yml`, we have the following settings

1. Trigger Setting. The workflow automatically triggers on push requests to the dev branch and pull requests to the main branch.

```
on: 
  push:
    branches: [dev]
  pull_request:
    branches: [master]
```

2. Set up running environment and install essential packages

```
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.13'
- name: Install dependencies
  run: |
    python3 -m pip install --upgrade pip
    python3 -m pip install flake8 pytest pytest-asyncio ruff
    python3 -m pip install -r ./requirements.txt
```

3. Check flake 8 linter and execute all test suite

```
- name: Check style issue with Flake8
  run: |
    echo "Use Flake8"
    flake8 ./FastAPI_Backend/main.py
    flake8 ./Model_Management/train_model.py
    flake8 ./Monitor_Streamlit/monitor_app.py
    flake8 ./Streamlit_Frontend/frontend_app.py
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

- name: Run pytest on FastAPI_Backend
  run: |
    pytest -v ./FastAPI_Backend/test_backend.py

- name: Run pytest on Model_Management
  run: |
    pytest -v ./Model_Management/test_manage.py

- name: Run pytest on Monitor_Streamlit
  run: |
    pytest -v ./Monitor_Streamlit/test_dashboard.py

- name: Run pytest on Streamlit_Frontend
  run: |
    pytest -v ./Streamlit_Frontend/test_frontend.py
```

4. We create **requirements.txt** file in the `4705_Book_Purchase_Intention_Prediction` and implement Pytest to check code locally before pushing to Github.

  ```bash
  pip3 freeze > requirements.txt
  cp ./requirements.txt ./FastAPI_Backend
  cp ./requirements.txt ./Monitor_Streamlit
  cp ./requirements.txt ./Streamlit_Frontend
  cd FastAPI_Backend
  pytest -v test_backend.py
  cd ../Model_Management
  pytest -v test_manage.py
  cd ../Monitor_Streamlit
  pytest -v test_dashboard.py
  cd ../Streamlit_Frontend
  pytest -v test_frontend.py
  cd ..
  ```

5. Then, we run these commands to push all code to git repo.

```bash
git init
git add README.md
git commit -m "Initial commit"
git remote add origin https://github.com/jiansfoggy/4705_Personalized_Book_Recommender.git
git branch -M master
git push -u origin master
```

```bash
git status # make sure we are on master branch
git checkout -b dev
git status # make sure we are on dev branch
git add --all
git commit -m “init”
git push
```

6. Go to github and set up pull request.

    1. Click **Pull requests**
    2. Click **New pull request**
    3. Take `main` as base, `dev` as compare
    4. Click **Create pull request**
    5. Submit a new commit

7. Now, if everything goes well, you will see green check mark.
