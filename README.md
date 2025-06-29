## AI Development Workflow Assignment
## Description:
Assignment: Understanding the AI Development Workflow
Course: AI for Software Engineering
Duration: 7 days
Total Points: 100
---
## Assignment Overview
This project is part of a peer-group assignment for the PLP Academy AI for Software Engineering program. It demonstrates the application of the AI Development Workflow to a real-world problem. The goal is to showcase a full-cycle understanding of AI—from problem definition, data preparation, model training, and evaluation to deployment—while critically analyzing practical challenges and ethical considerations.
## 📂 Project Structure

```

📁 project-root/
│
├── 📁 data/                  # Contains mock patient dataset (patients.csv)
│
├── 📁 models/                # Saved ML models (.joblib)
│
├── 📁 src/                   # Source code
│   ├── app.py               # FastAPI app with /predict endpoint
│   ├── train\_model.py       # ML training pipeline
│   ├── evaluate\_model.py    # Evaluation and metrics
│   ├── data\_loader.py       # Data preprocessing utility
│
├── 📄 requirements.txt       # Python package dependencies
│
├── 📄 part1.pdf              # Part 1: Short Answer Questions
├── 📄 part2.pdf              # Part 2: Case Study Application
├── 📄 part3.pdf              # Part 3: Critical Thinking
├── 📄 part4.pdf              # Part 4: Reflection & Workflow Diagram

````

---

## 🚀 How to Run the Project

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
````

2. **Train the model**

   ```bash
   python src/train_model.py
   ```

3. **Evaluate the model**

   ```bash
   python src/evaluate_model.py
   ```

4. **Run the FastAPI app**

   ```bash
   uvicorn src.app:app --reload
   ```

5. **Test the API**
   Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   or use `curl`/Postman to send POST requests to `/predict`.

---

## 📈 Model Info

* Model: `XGBoostClassifier`
* Accuracy: 75%
* High recall for class 1 (positive readmissions)
* Balanced preprocessing and robust feature handling

---

## 👥 Team Members (Peer Group Project)

* **Scholar Waweru**  – 📧 *[scholarwambui23@gmail.com](mailto:scholarwambui23@gmail.com)*
* **Sharon Kipsang**     – 📧 *[sharonkipsang53@gmail.com](mailto:sharonkipsang53@gmail.com)*
* **Pauline Onyango** – 📧 *[paulineakoth2002@gmail.com](mailto:paulineakoth2002@gmail.com)*

---

## 📌 Notes

* Built with educational goals in mind.
* Sample dataset (`patients.csv`) is simulated.
* Real-world deployments must ensure **HIPAA** or **Data Protection Act** compliance.