# CustomGPT

CustomGPT is a custom implementation of a GPT-like model designed for training on text datasets, tokenization with a Byte Pair Encoding (BPE) approach, and inference for tasks such as question-answering. The project includes modules for data preprocessing, model training, inference, and user interfaces.

## Project Structure

- **tokenizer.py**  
  Contains the `BPETokenizer` class used for tokenizing text. It implements methods for fitting a vocabulary on a corpus, applying merges, encoding, and decoding tokens.  
  Refer to the [tokenizer module](./tokenizer.py) for more details.

- **dataset.py**  
  Implements the `WikiDataset` class and a custom `collate_fn` for preparing data for training.

- **model.py**  
  Defines the `CustomGPT` model architecture and model hyperparameters.

- **train.py & train_bpe.py**  
  Contains training loops and functions (`train_gpt`) used to train the CustomGPT model as well as the BPE tokenizer.

- **main.py**  
  Acts as a central entry-point for training the model on different datasets, including options for data loading, tokenization, and setting up training pipelines.

- **inference.py & generate.py**  
  Provides functions (e.g., `generate_text`) for autoregressively generating text from a given prompt using the trained model.

- **question_answer.py**  
  Offers functionality to create a QA prompt and tokenize data for training on question-answering datasets.

- **interface.py**  
  Includes UI interfaces (with Gradio/Streamlit) for interacting with the model via chat or basic Q&A functionalities.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://your-repository-url.git
   cd CustomGPT
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**
   Place your datasets (e.g., Wikitext-2, BBC News, QA data) in the `data/` folder. Adjust file paths in the source code if needed.

## Usage

### Training the Model

- **Training Custom GPT Model**  
  Run the main training script:
  ```bash
  python main.py
  ```
  This script loads the data, tokenizes the text using the BPE tokenizer, and trains the CustomGPT model.

- **Training Tokenizer (BPE)**
  To build and save a new vocabulary, use:
  ```bash
  python train_bpe.py
  ```

- **Question-Answering Training**  
  Use the `question_answer.py` script for training on QA-specific datasets:
  ```bash
  python question_answer.py
  ```

### Inference & Text Generation

- **Generate Text**  
  Use the `inference.py` or `generate.py` scripts to generate text from a prompt. For example:
  ```bash
  python generate.py
  ```

- **Interactive Interface**  
  Launch the user interface (Gradio/Streamlit) for chatting:
  ```bash
  python interface.py
  ```

## Customization

- **Tokenizer**  
  Modify or extend [`BPETokenizer`](./tokenizer.py) methods such as `_apply_merges` for different tokenization strategies.

- **Model Architecture**  
  Customize hyperparameters (e.g., `context_length`, `embed_dim`, `hidden_dim`, `num_heads`, `num_layers`) in [`main.py`](./main.py) or directly within the `CustomGPT` model in the [`model.py`](./model.py).

- **Training Loop**  
  Adjust training behavior (e.g., learning rate, scheduler, loss function) in [`train.py`](./train.py).

## Requirements

- Python 3.8+
- PyTorch
- Pandas
- scikit-learn
- transformers (for LR scheduler)
- Gradio / Streamlit (for interfaces)
- Other dependencies listed in `requirements.txt`

## License

This project is licensed under the MIT License.

## Acknowledgements

This project builds on the ideas behind GPT models and uses custom tokenization methods inspired by Byte-Pair Encoding (BPE).

---

Happy coding!