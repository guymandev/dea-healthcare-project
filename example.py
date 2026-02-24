import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

class DataLake:
    def __init__(self, db_file='datalake.db'):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.create_table()


    def create_table(self):
        query = '''
            CREATE TABLE IF NOT EXISTS data(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                value FLOAT            
            );
        '''

        self.conn.execute(query)
        self.conn.commit()


    def collect_data(self, value):
        query = 'INSERT INTO data (value) VALUES (?);'
        self.conn.execute(query, (value,))  # <-- note the comma
        self.conn.commit()

    def retrieve_data(self):
        query = 'SELECT * FROM data;'
        data = pd.read_sql_query(query, self.conn)
        return data
    
    def analyze_data(self):
        data = self.retrieve_data()
        mean_value = data['value'].mean()
        std_dev = data['value'].std()
        return mean_value, std_dev
    
    def visualize_data(self):
        data = self.retrieve_data()
        plt.plot(data['timestamp'], data['value']) 
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.title('Data Lake Visualization')
        plt.show()

def main():
    data_lake = DataLake()
    data_lake.collect_data(5.5)
    data_lake.collect_data(7.2)

    retrieved_data = data_lake.retrieve_data()
    print("Retrieved Data is : ")
    print(retrieved_data)

    mean, std_dev = data_lake.analyze_data()
    print("\nData Analysis : ")
    print(f'Mean Value : {mean}')
    print(f'Standard Deviation: {std_dev}')

    data_lake.visualize_data()

if __name__ == "__main__":
    main()