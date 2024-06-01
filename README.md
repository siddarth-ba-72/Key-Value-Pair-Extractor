# Key Value Pair Extraction using LLM

## Step 1: Create an Environment
```
python -m venv env
```

## Step 2: Install Dependencies
```
pip install -r requirements.txt
```

## Step 3: Keep your document/file in the `/data` folder

## Step 4: Prepare Database
```
python populate_database.py --reset
```

## Step 5: Give the prompt to get the output
```
python query_data.py <query/prompt>
```
