import torch
from dataset import Vocabulary
from model import HarryPotterTransformer


def generate(model, device, seed_words, input_length, output_length, vocab, temperature=1.0, strategy='sampling'):
    """
    Arguments:
        model: HarryPotterTransformer
        device: string, presents tensor's device
        seed_words: string
        input_length: int, model's max input length
        output_length: int, max output length during generation
        vocab: Vocabulary, defined in dataset.py
        temperature: float, temperature during model inference
        strategy: string, should be either 'sampling' based on model output probabilities 
                  or 'greedy' by choosing the maximum probability. 
    """

    model.eval()

    with torch.no_grad():

        output_arr = torch.LongTensor([-1]).to(device)

        ################################################################################
        # TODO: generate a paragraph from seed_words                                   #
        # To complete this task, you need to review the input and output formats       #
        # of various functions and models in other code file, and integrate them to    #
        # achieve our goal.                                                            #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x = vocab.words_to_array(seed_words).to(device)
        output_arr = x

        for _ in range(output_length):
            x = model.inference(output_arr[-input_length:])

            if strategy == 'greedy':
                x = x.argmax(dim=-1)
            elif strategy == 'sampling':
                x = torch.multinomial(x.squeeze(), 1)
            else:
                raise ValueError("Strategy should be either 'sampling' or 'greedy'.")

            output_arr = torch.concat((output_arr, x), 0)

            if x.item() == vocab.words_to_array("#").item():
                break


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return vocab.array_to_words(output_arr[:-1]), len(output_arr)-1

# hyper-parameters, shouble be same as in main.py
SEQUENCE_LENGTH = 100
FEATURE_SIZE = 512
NUM_HEADS = 8
USE_CUDA = True

MODEL_TYPE = 'transformer'
# DATA_PATH = 'data/'
# EXP_PATH = f'exp/'
DATA_PATH = '/content/drive/MyDrive/programming3/data/'
EXP_PATH = f'/content/drive/MyDrive/programming3/exp/'

if __name__=="__main__":
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    # build vocab
    vocab = Vocabulary(DATA_PATH + 'harry_potter_chars_train.pkl')

    # build model
    if MODEL_TYPE == 'transformer':
        model = HarryPotterTransformer(len(vocab), FEATURE_SIZE, NUM_HEADS).to(device)
        model.load_last_model(EXP_PATH + 'checkpoints/')
    else:
        raise NotImplementedError

    # generate
    # Try different seed_words. Find interesting ones.
    seed_words = "Harry Potter "
    # Experiment with different temperatures to observe their impact on diversity and stability.
    temperature = 1.0
    # Experiment with different strategies.
    strategy = 'sampling'
    #strategy = 'greedy'

    generated_sentence, array_length = generate(model, device, seed_words, SEQUENCE_LENGTH, 2000, vocab, temperature, strategy)
    print("Generated:\n", generated_sentence)
    print("Generated Length:", array_length)