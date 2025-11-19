import numpy as np
from html_logger import HtmlLogger
import os

def reproduce():
    logger = HtmlLogger()
    
    # Simulate moves as numpy integers, which is what MCTSPlayer returns
    moves = [np.int64(i) for i in range(10)]
    winner = 1
    
    try:
        logger.save_game(moves, winner, filename="test_log.html")
        print("Successfully saved game log.")
        
        # Check the content of the file
        with open("logs/test_log.html", "r") as f:
            content = f.read()
            print("Content snippet:")
            print(content[:200])
            
            # Check if moves are correctly formatted in JS
            if "const moves = [" in content:
                print("Moves array looks correct.")
            else:
                print("Moves array might be malformed.")
                # Find the line with const moves
                for line in content.splitlines():
                    if "const moves =" in line:
                        print(f"Found line: {line}")
                        
    except Exception as e:
        print(f"Error saving game log: {e}")

if __name__ == "__main__":
    reproduce()
