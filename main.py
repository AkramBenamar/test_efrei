from time import time
import os
from data.data_processing import DataPreprocessor
from models.siamese_model import SiameseLSTM
import json
def main(data_directory, train_csv, max_seq_length, sample_size, n_epoch, batch_size):
    # Initialize preprocessor
    embedding_path = os.path.join(data_directory, 'GoogleNews-vectors-negative300.bin.gz')
   
    preprocessor = DataPreprocessor(embedding_path, sample_size)

    # Load and process data
    X_train, X_validation, Y_train, Y_validation,X_test, Y_test, embeddings = preprocessor.load_and_process_data(train_csv, max_seq_length)

    # Initialize and build model
    siamese_model = SiameseLSTM(embeddings, embedding_dim=300, max_seq_length=max_seq_length)
    model = siamese_model.build_model()

    # Summary
    model.summary()

    # Train model
    training_start_time = time()

    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation))

    training_end_time = time()
    history_dict = malstm_trained.history
    with open('checkpoints/training_history_siameseLstm.json', 'w') as f:
        json.dump(history_dict, f)
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    # Save model
    model.save('SiameseLSTM.h5')

    # Evaluate model
    evaluation_results = model.evaluate([X_test['left'], X_test['right']], Y_test, verbose=1)
    print(f"Test Loss: {evaluation_results[0]}, Test Accuracy: {evaluation_results[1]}")

if __name__ == "__main__":
    data_directory = 'C:/Users/DELL/Desktop/Akm/test_technique_efrei/CompAIre\data'
    train_csv = os.path.join(data_directory, 'questions.csv')
    # train_csv='/content/drive/MyDrive/dataset_efrei_test/questions.csv'
    max_seq_length = 20
    sample_size = 10000
    n_epoch = 50
    batch_size = 2048

    main(data_directory, train_csv, max_seq_length, sample_size, n_epoch, batch_size)
