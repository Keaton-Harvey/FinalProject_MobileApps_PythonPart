# Blackjack Game for data collection
import pandas as pd
import random

random.seed(42)

# Define card suits and ranks
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, 'Jack': 10, 'Queen': 10,
    'King': 10, 'Ace': 11  # Ace is counted as 11 initially
}

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f'{self.rank} of {self.suit}'

    def value(self):
        return ranks[self.rank]

class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        random.shuffle(self.cards)

    def deal_card(self):
        if len(self.cards) == 0:
            # Reinitialize the deck if all cards have been dealt
            self.__init__()
        return self.cards.pop()

class Hand:
    def __init__(self):
        self.cards = []
        self.total = 0
        self.soft_aces = 0  # Number of aces counted as 11

    def add_card(self, card):
        self.cards.append(card)
        self.total += card.value()
        if card.rank == 'Ace':
            self.soft_aces += 1  # Initially count Ace as 11
        self.adjust_for_ace()

    def adjust_for_ace(self):
        # Adjust for aces if total is over 21
        while self.total > 21 and self.soft_aces > 0:
            self.total -= 10  # Counting one Ace as 1 instead of 11
            self.soft_aces -= 1

    def is_soft(self):
        # A hand is soft if there are any aces counted as 11
        return self.soft_aces > 0

    def can_split(self):
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    def __str__(self):
        return ', '.join(str(card) for card in self.cards)

# Basic strategy based on standard blackjack strategy charts
def basic_strategy(player_hand, dealer_upcard_value):
    player_total = player_hand.total
    dealer_card = dealer_upcard_value
    is_soft = player_hand.is_soft()
    can_split = player_hand.can_split()
    
    # Pair Splitting Strategy
    if can_split:
        pair_rank = player_hand.cards[0].rank
        # Map face cards to '10'
        if pair_rank in ['Jack', 'Queen', 'King']:
            pair_rank = '10'
        elif pair_rank == 'Ace':
            pair_rank = 'A'

        # Define splitting decisions
        split_decisions = {
            'A': 'Split',
            '8': 'Split',
            '10': 'Stand',
            '9': 'Split' if dealer_card not in [7, 10, 11] else 'Stand',
            '7': 'Split' if dealer_card in range(2, 8) else 'Hit',
            '6': 'Split' if dealer_card in range(2, 7) else 'Hit',
            '5': 'Double Down' if dealer_card in range(2, 10) else 'Hit',
            '4': 'Split' if dealer_card in [5, 6] else 'Hit',
            '3': 'Split' if dealer_card in range(2, 8) else 'Hit',
            '2': 'Split' if dealer_card in range(2, 8) else 'Hit',
        }

        action = split_decisions.get(pair_rank, 'Hit')
        return action
    
    # Soft Totals Strategy
    if is_soft:
        if player_total >= 19:
            return 'Stand'
        elif player_total == 18:
            if dealer_card in range(2, 9):  # Dealer shows 2-8
                return 'Stand'
            elif dealer_card == 9 or dealer_card == 10 or dealer_card == 11:
                return 'Hit'
            else:
                return 'Hit'
        elif player_total == 17:
            if dealer_card in range(3, 7):  # Dealer shows 3-6
                return 'Double Down'
            else:
                return 'Hit'
        elif player_total in [15, 16]:
            if dealer_card in range(4, 7):  # Dealer shows 4-6
                return 'Double Down'
            else:
                return 'Hit'
        elif player_total in [13, 14]:
            if dealer_card in range(5, 7):  # Dealer shows 5-6
                return 'Double Down'
            else:
                return 'Hit'
        else:
            return 'Hit'
    
    # Hard Totals Strategy
    if player_total >= 17:
        return 'Stand'
    elif player_total == 16 or player_total == 15:
        if dealer_card in range(2, 7):  # Dealer shows 2-6
            return 'Stand'
        else:
            return 'Hit'
    elif player_total == 14 or player_total == 13:
        if dealer_card in range(2, 7):  # Dealer shows 2-6
            return 'Stand'
        else:
            return 'Hit'
    elif player_total == 12:
        if dealer_card in range(4, 7):  # Dealer shows 4-6
            return 'Stand'
        else:
            return 'Hit'
    elif player_total == 11:
        return 'Double Down'
    elif player_total == 10:
        if dealer_card in range(2, 10):  # Dealer shows 2-9
            return 'Double Down'
        else:
            return 'Hit'
    elif player_total == 9:
        if dealer_card in range(3, 7):  # Dealer shows 3-6
            return 'Double Down'
        else:
            return 'Hit'
    else:
        return 'Hit'

def play_blackjack(game_data):
    # Initialize deck and hands
    deck = Deck()
    player_hand = Hand()
    dealer_hand = Hand()

    # Deal initial cards
    player_hand.add_card(deck.deal_card())
    dealer_hand.add_card(deck.deal_card())
    player_hand.add_card(deck.deal_card())
    dealer_hand.add_card(deck.deal_card())

    # Get dealer's upcard value
    dealer_upcard_value = dealer_hand.cards[0].value()
    if dealer_hand.cards[0].rank == 'Ace':
        dealer_upcard_value = 11  # Ace is 11

    # Collect data for the initial hand
    action = basic_strategy(player_hand, dealer_upcard_value)
    game_state = {
        'Player Total': player_hand.total,
        'Dealer Upcard': dealer_upcard_value,
        'Has Ace': int(player_hand.is_soft()),  # Convert boolean to integer (0 or 1)
        'Recommended Action': action
    }
    game_data.append(game_state)

if __name__ == "__main__":
    # Initialize data collection list
    game_data = []

    # Number of simulations
    num_simulations = 750000

    # Run simulations
    for _ in range(num_simulations):
        play_blackjack(game_data)
    
    df = pd.DataFrame(game_data)
    df.to_csv('blackjack_data.csv', index=False)