import array

import numpy as np 
import pandas as pd

def random_position_generator(image : array.array[float],
                              nrand : int) -> array.array[float]:
    """
    Generate a random position within the image weighted by the pixel values.

    ARGS
    ====
    image :     Image to generate random positions from.
    nrand :     Number of random positions to generate.

    RETURNS
    =======
    rand_pos : 2D numpy array
        The random positions generated.
    """

    # turn the image into a dataframe
    df = pd.DataFrame(columns=['x', 'y', 'value'])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_row = {'x': i, 'y': j, 'value': image[i, j]}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # add column for cumulative sum of pixel
    df['cumul_sum'] = df['value'].cumsum()

    # get the sum of the pixel values
    total = df['value'].sum()

    # add a column for the probability of each pixel
    df['prob'] = df['cumul_sum'] / total

    # generate nrand random numbers between 0 and 1
    rand = np.random.rand(nrand)

    # find the pixel that corresponds to each random number
    rand_pos_x = []
    rand_pos_y = []

    for r in rand:
        pixel = df[df['prob'] >= r].iloc[0]
        delta_ran = np.random.rand(2)
        ran_sign_x = np.random.choice([-1, 1])
        ran_sign_y = np.random.choice([-1, 1])
        rand_pos_x.append(pixel['x'] + ran_sign_x * delta_ran[0])
        rand_pos_y.append(pixel['y'] + ran_sign_y * delta_ran[1])

    rand_pos = np.array([rand_pos_x, rand_pos_y]).T

    return rand_pos