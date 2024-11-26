import os
import sys
import winreg

def setup_hadoop_env():
    # Create HADOOP_HOME environment variable
    hadoop_home = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'hadoop')
    os.makedirs(hadoop_home, exist_ok=True)
    
    # Set environment variables
    os.environ['HADOOP_HOME'] = hadoop_home
    os.environ['PATH'] = f"{hadoop_home}\\bin;{os.environ['PATH']}"
    
    print(f"HADOOP_HOME set to: {hadoop_home}")
    return hadoop_home

if __name__ == "__main__":
    hadoop_home = setup_hadoop_env() 