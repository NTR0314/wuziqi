import os

class HtmlLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def save_game(self, moves, winner, filename="game_log.html"):
        """
        moves: list of move indices (0-224)
        winner: player index (1 or 2, or -1 for tie)
        """
        path = os.path.join(self.log_dir, filename)
        
        # Ensure moves are standard python ints for JSON serialization in JS
        moves = [int(m) for m in moves]
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Gomoku Game Log</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; background-color: #f0f0f0; }}
        #board {{ 
            width: 600px; 
            height: 600px; 
            background-color: #eebb77; 
            margin: 20px auto; 
            position: relative; 
            border: 2px solid #000;
            display: grid;
            grid-template-columns: repeat(15, 1fr);
            grid-template-rows: repeat(15, 1fr);
        }}
        .cell {{
            border: 1px solid rgba(0,0,0,0.2);
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .stone {{
            width: 80%;
            height: 80%;
            border-radius: 50%;
            box-shadow: 2px 2px 2px rgba(0,0,0,0.3);
            display: none;
        }}
        .stone.black {{ background-color: #000; display: block; }}
        .stone.white {{ background-color: #fff; display: block; }}
        .stone.last {{ border: 2px solid red; box-sizing: border-box; }}
        .controls {{ margin: 20px; }}
        button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>Gomoku Game Log</h1>
    <h2 id="status">Winner: {winner if winner != -1 else "Tie"}</h2>
    <div id="board"></div>
    <div class="controls">
        <button onclick="prevMove()">Previous</button>
        <button onclick="nextMove()">Next</button>
        <span id="move-count">Move: 0</span>
    </div>

    <script>
        const moves = {moves};
        let currentMoveIndex = -1;
        const boardEl = document.getElementById('board');
        const moveCountEl = document.getElementById('move-count');
        const cells = [];

        // Init board
        for (let i = 0; i < 225; i++) {{
            const cell = document.createElement('div');
            cell.className = 'cell';
            const stone = document.createElement('div');
            stone.className = 'stone';
            cell.appendChild(stone);
            boardEl.appendChild(cell);
            cells.push(stone);
        }}

        function updateBoard() {{
            // Clear all
            cells.forEach(s => {{ s.className = 'stone'; }});

            // Apply moves up to current
            for (let i = 0; i <= currentMoveIndex; i++) {{
                const move = moves[i];
                const color = (i % 2 === 0) ? 'black' : 'white';
                cells[move].classList.add(color);
                if (i === currentMoveIndex) {{
                    cells[move].classList.add('last');
                }}
            }}
            moveCountEl.innerText = 'Move: ' + (currentMoveIndex + 1);
        }}

        function nextMove() {{
            if (currentMoveIndex < moves.length - 1) {{
                currentMoveIndex++;
                updateBoard();
            }}
        }}

        function prevMove() {{
            if (currentMoveIndex >= 0) {{
                currentMoveIndex--;
                updateBoard();
            }}
        }}
        
        // Show end state initially
        currentMoveIndex = moves.length - 1;
        updateBoard();
    </script>
</body>
</html>
        """
        
        with open(path, "w") as f:
            f.write(html_content)
        print(f"Game log saved to {path}")
