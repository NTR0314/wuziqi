import unittest
from game import Board

class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board(width=15, height=15, n_in_row=5)
        self.board.init_board()

    def test_horizontal_win(self):
        moves = [0, 15, 1, 16, 2, 17, 3, 18, 4]
        # Player 1: 0, 1, 2, 3, 4 (Row 0)
        # Player 2: 15, 16, 17, 18 (Row 1)
        for m in moves:
            self.board.do_move(m)
        win, winner = self.board.has_a_winner()
        self.assertTrue(win)
        self.assertEqual(winner, self.board.players[0])

    def test_vertical_win(self):
        # Column 0
        moves = [0, 1, 15, 2, 30, 3, 45, 4, 60]
        # Player 1: 0, 15, 30, 45, 60
        for m in moves:
            self.board.do_move(m)
        win, winner = self.board.has_a_winner()
        self.assertTrue(win)
        self.assertEqual(winner, self.board.players[0])

    def test_diagonal_win(self):
        # (0,0), (1,1), (2,2), (3,3), (4,4)
        # 0, 16, 32, 48, 64
        moves = [0, 1, 16, 2, 32, 3, 48, 4, 64]
        for m in moves:
            self.board.do_move(m)
        win, winner = self.board.has_a_winner()
        self.assertTrue(win)
        self.assertEqual(winner, self.board.players[0])

    def test_anti_diagonal_win(self):
        # (0,4), (1,3), (2,2), (3,1), (4,0)
        # 4, 18, 32, 46, 60
        moves = [4, 0, 18, 1, 32, 2, 46, 3, 60]
        for m in moves:
            self.board.do_move(m)
        win, winner = self.board.has_a_winner()
        self.assertTrue(win)
        self.assertEqual(winner, self.board.players[0])

    def test_no_win(self):
        moves = [0, 1, 2, 3]
        for m in moves:
            self.board.do_move(m)
        win, winner = self.board.has_a_winner()
        self.assertFalse(win)

if __name__ == '__main__':
    unittest.main()
