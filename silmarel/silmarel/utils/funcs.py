import array

import numpy as np 
import pandas as pd

def random_position_generator(image, nrand):
    """
    Generate a random position within the image weighted by the pixel values.

    Parameters
    ----------
    image : 2D numpy array
        The image to generate random positions from.
    nrand : int
        The number of random positions to generate.

    Returns
    -------
    rand_pos : 2D numpy array
        The random positions generated.
    """

    # turn the image into a dataframe
    df = pd.DataFrame(columns=['x', 'y', 'value'])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_row = {'x': i, 'y': j, 'value': image[i, j]}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # get the sum of the pixel values
    total = df['value'].sum()
    df = df.sort_values(by = 'value', ignore_index=True, ascending=False)

    # add column for cumulative sum of pixel 
    df['cumul_sum'] = df['value'].cumsum()

    # add a column for the probability of each pixel
    df['prob'] = df['cumul_sum'] / total

    # generate nrand random numbers between 0 and 1
    rand = np.random.rand(nrand)

    # find the pixel that corresponds to each random number
    rand_pos_x = []
    rand_pos_y = []

    for r in rand:
        pixel = df[df['prob'] >= r].iloc[0]
        xmin = pixel['x']
        ymin = pixel['y']
        rand_pos_x.append(np.random.uniform(xmin-0.5, xmin+0.5))
        rand_pos_y.append(np.random.uniform(ymin-0.5, ymin+0.5))

    rand_pos = np.array([rand_pos_x, rand_pos_y]).T

    return rand_pos