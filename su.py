class Submission:
    def __init__(self, board_size, win_size):
        self.board_size = board_size
        self.win_size = win_size
        self.max_depth = 4  # You can adjust this value to control the search depth

    def __call__(self, state):
        _, action = self.minimax(state, self.max_depth)
        return action

    def minimax(self, state, max_depth, alpha=-np.inf, beta=np.inf):
        score, action = self.look_ahead(state)
        if score != 0:
            return score, action

        if state.is_game_over():
            return state.current_score(), None

        actions = state.valid_actions()
        rank = -state.corr[:, 1:].sum(axis=(0, 1)) - np.random.rand(*state.board.shape[1:])
        rank = rank[state.board[gm.EMPTY] > 0]
        scrambler = np.argsort(rank)

        if max_depth == 0:
            return state.current_score(), actions[scrambler[0]]

        if self.turn_bound(state) > max_depth:
            return 0, actions[scrambler[0]]

        best_action = None
        if state.is_max_turn():
            bound = -np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax(child, max_depth - 1, alpha, beta)

                if utility > bound:
                    bound, best_action = utility, action
                if bound >= beta:
                    break
                alpha = max(alpha, bound)

        else:
            bound = +np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax(child, max_depth - 1, alpha, beta)

                if utility < bound:
                    bound, best_action = utility, action
                if bound <= alpha:
                    break
                beta = min(beta, bound)

        return bound, best_action

    def look_ahead(self, state):
        player = state.current_player()
        sign = +1 if player == gm.MAX else -1
        magnitude = state.board[gm.EMPTY].sum()

        corr = state.corr
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size - 1))
        if idx.shape[0] > 0:
            p, r, c = idx[0]
            action = self.find_empty(state, p, r, c)
            return sign * magnitude, action

        opponent = gm.MIN if state.is_max_turn() else gm.MAX
        loss_empties = set()
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))
        for p, r, c in idx:
            pos = self.find_empty(state, p, r, c)
            loss_empties.add(pos)
            if len(loss_empties) > 1:
                score = -sign * (magnitude - 1)
                return score, pos

        return 0, None

    def find_empty(self, state, p, r, c):
        if p == 0:
            return r, c + state.board[gm.EMPTY, r, c:c + state.win_size].argmax()
        if p == 1:
            return r + state.board[gm.EMPTY, r:r + state.win_size, c].argmax(), c
        if p == 2:
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
            return r + offset, c + offset
        if p == 3:
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
            return r - offset, c + offset
        return None

    def turn_bound(self, state):
        is_max = state.is_max_turn()
        fewest_moves = state.board[gm.EMPTY].sum()

        corr = state.corr
        min_routes = (corr[:, gm.EMPTY] + corr[:, gm.MIN] == state.win_size)
        max_routes = (corr[:, gm.EMPTY] + corr[:, gm.MAX] == state.win_size)
        min_turns = 2 * corr[:, gm.EMPTY] - (0 if is_max else 1)
        max_turns = 2 * corr[:, gm.EMPTY] - (1 if is_max else 0)

        if min_routes.any():
            moves_to_win = min_turns.flatten()[min_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)
        if max_routes.any():
            moves_to_win = max_turns.flatten()[max_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)

        return fewest_moves
