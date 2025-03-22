### **Wealth Asset Management Project**

---

#### **Overview**
This project is designed to support a Wealth Asset Management platform by providing a structured data solution to manage clients' portfolios, assets, transactions, and perform risk assessments. The project also showcases a **Text-to-SQL model** for querying the database and uses **LangChain** to assist in generating dynamic queries. The data is stored in **Microsoft Fabric Lakehouse**, ensuring scalability and performance for large data sets, making this solution suitable for financial institutions, wealth management firms, and analysts.

The entire workflow involves setting up a Microsoft Fabric Lakehouse, inserting 1GB of dummy data into the tables, and demonstrating how a Text-to-SQL model can be used to query this data.

---

### **Microsoft Fabric Lakehouse**

**Microsoft Fabric Lakehouse** is a central part of this project as it provides the cloud-based storage layer for all the data related to clients, portfolios, assets, transactions, and risk assessments. The Lakehouse enables efficient querying and reporting, which is critical for the performance of real-time dashboards and analysis.

After setting up the database schema (which includes tables, views, and stored procedures), the next step is to populate the database with the dummy data required for generating insights. The data inserted into this Lakehouse forms the basis for the wealth management operations, including risk assessments, asset allocations, and transaction tracking.

---

### **Data Insertion**

Before running any analysis, we need to insert data into the Microsoft Fabric Lakehouse. The project includes a Python script designed to insert 1GB of dummy data into the database. This will simulate real-world scenarios by populating the tables with enough data for meaningful visualization and analytical insights.

#### **File to Insert Data:**
- `data_insertion.py` - This Python file contains the logic to connect to the Microsoft Fabric Lakehouse and insert dummy data into the database.

#### **Command to Run Data Insertion:**
```bash
python CreateDataWarehouse/insertToSQL.py
```

Ensure that you have the required packages installed (`pyodbc`, `faker`, etc.) and have updated the connection string in the script to connect to your Microsoft Fabric Lakehouse.

---

### **Text-to-SQL Model**

The **Text-to-SQL model** is a crucial part of this project. This AI-driven model is trained to convert natural language questions into SQL queries, making it easier for users who are not familiar with SQL to interact with the wealth management data.

This model is pre-trained and fine-tuned to understand industry-specific queries. You can use it to:
- Query portfolio values,
- Analyze asset distributions,
- Retrieve transaction histories,
- Assess risks, and more.

The Text-to-SQL model dynamically generates SQL queries based on the user’s input and runs those queries against the data stored in the Microsoft Fabric Lakehouse.

#### **File to Run Text-to-SQL Model:**
- Train Text-to-SQL Model: `RAGToSQL/TrainRAG.py`
- Inference Text-to-SQL Model: `RAGToSQL/InferenceRAG.py`
- Fabrics Text-to-SQL Model: `RAGToSQL/InferenceRAG.py`


---

### **LangChain Integration**

We use **LangChain** to further enhance the capabilities of the Text-to-SQL model by adding an intelligent context layer. LangChain helps in managing dynamic prompts and contextual conversations with the user, which can be used to refine queries or generate additional insights.

This integration allows the model to handle complex queries, perform multi-step reasoning, and even conduct follow-up questions based on prior results. LangChain ensures that the interaction feels natural and the queries remain focused on the desired insights.

#### **File to Run LangChain Integration:**
- LangChain with Fabrics : `LangChainFabrics.py`

---

### **Project Structure**

- **`CreateDataWarehouse`**: Contains the SQL Server scripts for creating tables, views, and stored procedures. Python script to insert 1GB of dummy data into the Microsoft Fabric Lakehouse.
- **`RAGToSQL`**: Contains files to train and inference RAGToSQL 
- **`LangchainFabrics`**: Script for running the Text-to-SQL model with Langchain Vector DB.
- **`requirements.txt`**: Contains a list of required Python packages to run the project.

---

### **Prerequisites**

1. **Python 3.9**
2. **SQL Server & Microsoft Fabric Lakehouse Access**
3. **Required Libraries**: Install dependencies via `pip install -r requirements.txt`

### **Setup Instructions**

1. **Set Up Microsoft Fabric Lakehouse**: Ensure that the lakehouse is set up and ready to accept the schema. 
2. **Insert Data**: Run the `CreateDataWarehouse` script to populate the tables with dummy data.
3. **Run Text-to-SQL Model**: Interact with the wealth asset management data using the Text-to-SQL model for dynamic query generation.
4. **Leverage LangChain**: Use LangChain for more advanced queries and conversational interaction with the data.


# Execution Instructions

## Python version 3.9

To create a virtual environment and install requirements in Python 3.9 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing `Win + R`, typing `cmd`, and pressing `Enter`.

Change the directory to the desired location for your project:

```sh
cd C:\path\to\project
```

Create a new virtual environment using the `venv` module:

```sh
python -m venv myenv
```

Activate the virtual environment:

```sh
myenv\Scripts\activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

### For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

```sh
cd /path/to/project
```

Create a new virtual environment using the `venv` module:

```sh
python3.9 -m venv myenv
```

Activate the virtual environment:

```sh
source myenv/bin/activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

These instructions assume you have Python 3.9 installed and added to your system's `PATH` variable.

## Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions installed on your system, you can use the Python Launcher to create a virtual environment with Python 3.9. Specify the version using the `-p` or `--python` flag. Follow the instructions below:

### For Windows:

Open the Command Prompt by pressing `Win + R`, typing `cmd`, and pressing `Enter`.

Change the directory to the desired location for your project:

```sh
cd C:\path\to\project
```

Create a new virtual environment using the Python Launcher:

```sh
py -3.9 -m venv myenv
```

> **Note**: Replace `myenv` with your desired virtual environment name.

Activate the virtual environment:

```sh
myenv\Scripts\activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

### For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

```sh
cd /path/to/project
```

Create a new virtual environment using the Python Launcher:

```sh
python3.9 -m venv myenv
```

> **Note**: Replace `myenv` with your desired virtual environment name.

Activate the virtual environment:

```sh
source myenv/bin/activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

By specifying the version using `py -3.9` or `python3.9`, you can ensure that the virtual environment is created using Python 3.9 specifically, even if you have other Python versions installed.

## Additional Setup Instructions

1. Run `connection.py` to create Azure credentials by authenticating your account. This will print a token in the terminal.
2. Copy the token and paste it into the `credentials.py` script located in `helper` within the `RAGtoSQL` folder in `financial_goals`.
3. Ensure all credentials are working correctly.
4. Run `app.py` in the root folder to start the application. using 
`python app.py`
``
├─ Agent
│  ├─ agent.py
│  ├─ intent.py
│  ├─ memory.py
│  ├─ tools.py
├─ app.py
├─ connection.py
├─ CreateDataWarehouse
│  ├─ CreatingTables.sql
│  └─ InsertToSQL.py
├─ FinancialGoals
│  └─ RAGToSQL
│     ├─ FabricsRAG.py
│     ├─ Helper
│     │  ├─ Credentials.py
│     │  ├─ FabricsConnection.py
│     │  ├─ VannaObject.py
│     ├─ InferenceRAG.py
│     ├─ TrainingRAG-Artifact
│     │  ├─ Documentation.txt
│     │  └─ Tables.json
│     ├─ training_summary.csv
│     ├─ TrainRAG.py
│     ├─ VisualizeRAG.py
├─ readme.md
├─ requirements.txt
└─ templates
   └─ index.html

```