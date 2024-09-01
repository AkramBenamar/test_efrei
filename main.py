from time import time
import os
from data.data_processing import DataPreprocessor
from models.siamese_model import SiameseLSTM
from models.attention_siamese_model import AttenSiameseLSTM
import json
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Siamese LSTM model on a question dataset.")

    # Adding arguments
    parser.add_argument('--model', type=str, default='siamese_lstm',choices=['siamese_lstm','attention_siamese_lstm'],
                        help="Choose a model to train.")
    parser.add_argument('--data_directory', type=str, required=True,
                        help="Directory where the data is located.")
    
    parser.add_argument('--max_seq_length', type=int, default=20,
                        help="Maximum sequence length for padding/truncating sequences.")
    parser.add_argument('--sample_size', type=int, default=10000,
                        help="Number of samples to use for training.")
    parser.add_argument('--n_epoch', type=int, default=50,
                        help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, default=2048,
                        help="Batch size for training.")

    return parser.parse_args()
def main(args):
    train_csv = os.path.join(args.data_directory, 'questions.csv')
    # Initialize preprocessor
    embedding_path = os.path.join(args.data_directory, 'GoogleNews-vectors-negative300.bin.gz')
   
    preprocessor = DataPreprocessor(embedding_path, args.sample_size)

    # Load and process data
    X_train, X_validation, Y_train, Y_validation,X_test, Y_test, embeddings = preprocessor.load_and_process_data(train_csv, args.max_seq_length)

    # Initialize and build model
    
    if(args.model=='siamese_lstm'):
        siamese_model = SiameseLSTM(embeddings, embedding_dim=300, max_seq_length=args.max_seq_length)
        model = siamese_model.build_model()
        
    elif(args.model=='attention_siamese_lstm'):
        siamese_model = AttenSiameseLSTM(embeddings, embedding_dim=300, max_seq_length=args.max_seq_length)
        model = siamese_model.build_model()
    else :
        raise ValueError("Model not implemented")
    print(f'Model {args.model} built')

    # Summary
    model.summary()

    # Train model
    training_start_time = time()

    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=args.batch_size, epochs=args.n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation))

    training_end_time = time()
    history_dict = malstm_trained.history
    with open(f'checkpoints/training_history_{args.model}.json', 'w') as f:
        json.dump(history_dict, f)
    print("Training time finished.\n%d epochs in %12.2f" % (args.n_epoch, training_end_time - training_start_time))

    # Save model
    model.save(f'checkpoints/{args.model}.h5')

    # Evaluate model
    evaluation_results = model.evaluate([X_test['left'], X_test['right']], Y_test, verbose=1)
    with open(f'checkpoints/training_history_{args.model}.json', 'a') as f:
        f.write(f'\nTest Loss: {evaluation_results[0]} Test accuracy :{evaluation_results[1]}\n')
    print(f"Test Loss: {evaluation_results[0]}, Test Accuracy: {evaluation_results[1]}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
