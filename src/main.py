""" Runs the Program """

# imports
from coordinator import coordinate


# Main function
def main():
    # neural network configurations
    config = {
        'topologies' : [
            {
                'layers' : (20, 12, 4),
                'activation' : 'relu'
            },
            {
                'layers' : (20, 10, 6, 3),
                'activation' : 'logistic'
            },
            {
                'layers' : (20, 16, 12, 8, 4),
                'activation' : 'relu'
            },
            {
                'layers' : (20, 14, 10, 6),
                'activation' : 'logistic'
            },
            {
                'layers' : (20, 10),
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
