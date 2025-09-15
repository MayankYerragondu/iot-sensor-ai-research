from contact_alarm.one_class_svm import run as run_contact
from pir_alarm.isolation_forest import run as run_pir
from env_sensor.LSTM_model import run as run_env

def main():
    print("Starting contact alarm script...")
    run_contact()
    print("Starting PIR alarm script...")
    run_pir()
    print("Starting environment sensor script...")
    run_env()
    print("All scripts executed successfully!")

if __name__ == "__main__":
    main()
