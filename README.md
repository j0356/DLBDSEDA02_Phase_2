# DLBDSEDA02_Phase_2

## 📚 Pre-Requisites
- Python 3.12
- git
## 👨‍💻 Code Setup
Open a terminal, then paste the following commands in correct order.

Step 1 - Clone the project
```
git clone https://github.com/j0356/DLBDSEDA02_Phase_2.git
```
Step 2 - Create a python virtual environment
```
cd DLBDSEDA02_Phase_2

---Windows---
python -m venv env
.\env\Scripts\Activate.ps1

---MacOS/Linux---
python3 -m venv env
./env/Scripts/activate
```
Step 3 - Install the necessary dependecies
```
pip install -r requirements.txt
```
Step 4 - Run the code
```
python main.py
```

## 📊 Visualization

After running `main.py`, a folder named **`output`** will be created, where you can find the following PNG files:

* **`coherence_score.png`** $\rightarrow$ Depicts the coherence score ($C_v$) for the range of topics tested to find the **optimal number of topics** based on this score.
* **`topics_lda_bow.png`** $\rightarrow$ Visualizes what each topic is about based on its **most important words**, which are also displayed with their **weight score**.
* **`topics_lsa_tf-idf.png`** $\rightarrow$ The **same visualization as LDA** but using the LSA algorithm (which sometimes finds different/complementary patterns).

## 🏃‍♂️‍➡️ Runtime Info

The pipeline took **approximately 23 minutes** to run on the following CPU: **`AMD Ryzen 5 8400F 6-Core`**. However, this duration might differ based on the CPU you are running.

Additionally, the pipeline will take longer if you increase the number of topics to test for optimization, specifically, if you increase the **maximum range in `NUM_TOPICS_RANGE`**.