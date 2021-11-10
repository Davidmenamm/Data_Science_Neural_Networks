""" Runs the Program """

# imports
from coordinator import coordinate


# Main function
def main():
    # neural network configurations
    config = {
        'topologies' : [
            {
                'layers' : (16, 8, 6),
                'activation' : 'relu'
            },
            {
                'layers' : (30, 18, 10, 4),
                'activation' : 'logistic'
            },
            {
                'layers' : (24, 16, 12, 8, 2),
                'activation' : 'relu'
            },
            {
                'layers' : (36, 20, 6),
                'activation' : 'logistic'
            },
            {
                'layers' : (44, 20),
                'activation' : 'relu'
            }
        ],
        'hyperparameter' : {
            'lr' : [0.1, 0.3, 0.5],
            'epoch' : [500, 600, 700, 800, 900, 1000]
        }

    }
    # call coordinator
    coordinate(config = config, k_fold = 10, random_state = 100)


# if main function
if __name__ == "__main__":
    main()
