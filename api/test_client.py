"""
    Test requests sent to deployed APIs.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""

import requests


TEST_CASES = [
    'happy',
    'success',
    'well-being',
    'definitely',
    "didn't",
    'dejavu',
    "ash's",
    "chirped",
    "squatted",
    "coats"
]
URL = 'http://localhost:8000/infer_gps'




if __name__ == '__main__':

    # wrap up requests
    for word in TEST_CASES:
        response = requests.get(URL, params={'word': word})
        print(f"\nFor word [{word}], the inferred (grapheme, phoneme) pair is:\n"
              f"\t{response.json()}\n\n")