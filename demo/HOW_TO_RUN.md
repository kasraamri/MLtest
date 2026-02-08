# How to Run the PEP BALANCE AI Demo

## What You Need First

You need a computer with **Python** installed (version 3.9 or newer).

To check if you have Python, open your terminal and type:

```
python --version
```

If you see a number like `Python 3.10.4`, you're good. If not, download Python from https://www.python.org/downloads/ and install it.

---

## Step 1: Download the Project

Open your terminal (the black window where you type commands) and type:

```
git clone https://github.com/kasraamri/MLtest.git
```

Then go into the demo folder:

```
cd MLtest/demo
```

---

## Step 2: Install What the Demo Needs

Still in your terminal, type:

```
pip install -r requirements.txt
```

Wait until it finishes. You will see some text scrolling. When it stops and you can type again, it's done.

---

## Step 3: Run the Demo

Type:

```
python demo_app.py
```

That's it! The demo will:
1. Create fake employee data (takes less than 1 second)
2. Train a small AI model (takes less than 1 second)
3. Show you 4 examples of what the system can do

---

## What You Will See

The demo prints 4 scenarios to your screen:

**Scenario 1** - Finds employees who are marked as "available" but never actually get scheduled. Example: "Maria is available every Tuesday but was never scheduled — remove Tuesday from her availability."

**Scenario 2** - When a new task comes in, it ranks employees by who is the best fit. It shows scores, experience, and an AI confidence percentage.

**Scenario 3** - Finds employees who keep doing a task that isn't officially theirs. Example: "John keeps doing cleaning and is great at it — add it to his official tasks."

**Scenario 4** - Finds tasks assigned to employees that haven't been used in months. Example: "Maria hasn't done Special Events in 6 months — remove it from her assignments."

---

## Run Just One Scenario

If you only want to see one scenario, add `--scenario` and a number (1, 2, 3, or 4):

```
python demo_app.py --scenario 1
```

```
python demo_app.py --scenario 2
```

---

## Something Not Working?

| Problem | Fix |
|---|---|
| `python: command not found` | Try `python3 demo_app.py` instead |
| `pip: command not found` | Try `pip3 install -r requirements.txt` instead |
| `No module named pandas` | Run `pip install -r requirements.txt` again |
| Colors look weird | Your terminal might not support colors — the text content still works fine |
