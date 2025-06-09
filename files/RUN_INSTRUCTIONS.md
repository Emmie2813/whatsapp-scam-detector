# Step-by-Step Instructions: Uploading and Running Your WhatsApp Scam Detection Pipeline on GitHub

---

## 1. Prepare Your Files

- Unzip the files above into a folder (e.g., `scam_detector_project/`).
- Place your WhatsApp ZIPs in the `Data set/` folder inside this directory.

---

## 2. Create a New GitHub Repository

1. Go to [https://github.com/new](https://github.com/new) and create a new repository.
2. Give it a name (e.g., `whatsapp-scam-detector`).
3. Do **not** initialize with a README, .gitignore, or license.

---

## 3. Upload Your Project Files

**Option A: GitHub Web Interface**
- Click **Add file** > **Upload files**.
- Drag-and-drop all files and subfolders from your local project directory.

**Option B: Using Git**
- In your terminal, run:
    ```bash
    git init
    git remote add origin https://github.com/YourUsername/your-repo-name.git
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git push -u origin main
    ```

---

## 4. (Optional) Use GitHub Codespaces

- Click the green **Code** button > **Codespaces** > **Create codespace on main**.

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Run Your Pipeline

```bash
python extract_and_transcribe.py
python data_preprocessing.py
python train_model.py
python create_wordclouds.py
python app.py
```
Leave `app.py` running to serve the Flask API for your web app.

---

## 7. Connect Your Frontend

- Your web frontend should POST to `/analyze` on your Flask server.

---

## 8. Troubleshooting

- If you have errors, check the terminal output or ask for help!

---