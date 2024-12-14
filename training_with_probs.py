import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Action mapping
action_map = {"Hit":0, "Stand":1, "Double Down":2, "Split":3}

def card_value_distribution(num_decks):
    total_cards = 52 * num_decks
    probs = {
        '2': (4*num_decks)/total_cards,
        '3': (4*num_decks)/total_cards,
        '4': (4*num_decks)/total_cards,
        '5': (4*num_decks)/total_cards,
        '6': (4*num_decks)/total_cards,
        '7': (4*num_decks)/total_cards,
        '8': (4*num_decks)/total_cards,
        '9': (4*num_decks)/total_cards,
        '10': (16*num_decks)/total_cards, # 10,J,Q,K combined
        'A': (4*num_decks)/total_cards
    }
    return probs

def best_ace_value_for_draw(current_total):
    return 11 if current_total + 11 <= 21 else 1

def bust_probability_if_hit(player_total, num_decks):
    probs = card_value_distribution(num_decks)
    bust_prob = 0.0
    for card, p in probs.items():
        if card == 'A':
            card_val = best_ace_value_for_draw(player_total)
        else:
            card_val = int(card)
        if player_total + card_val > 21:
            bust_prob += p
    return bust_prob

def improve_hand_without_busting_if_hit(player_total, num_decks):
    # Probability that next card increases player's total without busting
    probs = card_value_distribution(num_decks)
    improve_prob = 0.0
    for card, p in probs.items():
        if card == 'A':
            card_val = best_ace_value_for_draw(player_total)
        else:
            card_val = int(card)
        new_total = player_total + card_val
        if new_total > player_total and new_total <= 21:
            improve_prob += p
    return improve_prob

def dealer_upcard_to_value(dealer_upcard):
    if dealer_upcard == 11:
        return 11
    else:
        return dealer_upcard

def if_stand_odds_dealer_second_card_beats_us(player_total, dealer_upcard, num_decks):
    # Probability dealer's next card alone surpasses player's total
    probs = card_value_distribution(num_decks)
    dealer_val = dealer_upcard_to_value(dealer_upcard)
    beat_prob = 0.0
    for card, p in probs.items():
        card_val = 11 if card=='A' else int(card)
        # If dealer_upcard=11 (Ace), check if need to count it as 1:
        # Actually with 2-card total:
        # If dealer_val=11(Ace), total_dealer=11+card_val. If >21, use Ace as 1 => total_dealer=1+card_val
        # If dealer_val=11:
        if dealer_val==11:
            if 11+card_val>21:
                total_dealer = 1+card_val
            else:
                total_dealer = 11+card_val
        else:
            # If no Ace upcard:
            total_dealer = dealer_val + card_val
            # If total_dealer>21 and we have an Ace card:
            # If card='A':
            if card=='A' and total_dealer>21:
                # Make Ace as 1
                total_dealer = dealer_val + 1

        if total_dealer > player_total:
            beat_prob += p

    return beat_prob

# Load data
data = pd.read_csv("perfect_blackjack_sim_data.csv")

data['action_code'] = data['best_action'].map(action_map)

bust_probs = []
improve_probs = []
dealer_beat_probs = []

for idx, row in data.iterrows():
    p_total = row['player_total']
    d_upcard = row['dealer_upcard']
    n_decks = row['num_decks']
    # dh_s17 = row['dealer_hits_soft_17'] # Not needed since we no longer simulate dealer draws, just second card.
    # Actually we don't use dealer_hits_soft_17 now?
    # The logic doesn't consider further dealer draws, only second card.
    # Let's keep it as a feature though.

    bp = bust_probability_if_hit(p_total, n_decks)
    ip = improve_hand_without_busting_if_hit(p_total, n_decks)
    dbp = if_stand_odds_dealer_second_card_beats_us(p_total, d_upcard, n_decks)

    bust_probs.append(bp)
    improve_probs.append(ip)
    dealer_beat_probs.append(dbp)

data['BustProbabilityIfHit'] = bust_probs
data['ImproveHandWithoutBustingIfHit'] = improve_probs
data['IfStandOddsDealersSecondCardMakesThemBeatUs'] = dealer_beat_probs

pair_rank_map = {'0':0,'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10}
if data['pair_rank'].dtype == object:
    data['pair_rank_encoded'] = data['pair_rank'].map(pair_rank_map)
else:
    data['pair_rank_encoded'] = data['pair_rank']

data.to_csv("training_with_probs_data.csv", index=False)

X = data[['player_total','dealer_upcard','is_soft','num_decks','dealer_hits_soft_17','can_split','pair_rank_encoded']]
y = data[['action_code','BustProbabilityIfHit','ImproveHandWithoutBustingIfHit','IfStandOddsDealersSecondCardMakesThemBeatUs']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

overall_score = model.score(X_test, y_test)
print(f"Overall R² Score on test set: {overall_score:.4f}")

target_names = ['action_code','BustProbabilityIfHit','ImproveHandWithoutBustingIfHit','IfStandOddsDealersSecondCardMakesThemBeatUs']
for i, name in enumerate(target_names):
    y_true_col = y_test.iloc[:, i]
    y_pred_col = y_pred[:, i]
    r2 = r2_score(y_true_col, y_pred_col)
    mae = mean_absolute_error(y_true_col, y_pred_col)
    mse = mean_squared_error(y_true_col, y_pred_col)
    rmse = np.sqrt(mse)
    print(f"\nMetrics for {name}:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")

print("\nSample of Actual vs Predicted (first 10 test samples):")
sample_size = min(10, len(y_test))
comparison_df = pd.DataFrame({
    'player_total': X_test.iloc[:sample_size]['player_total'],
    'dealer_upcard': X_test.iloc[:sample_size]['dealer_upcard'],
    'action_code_actual': y_test.iloc[:sample_size]['action_code'],
    'action_code_pred': y_pred[:sample_size, 0],
    'BustProbIfHit_actual': y_test.iloc[:sample_size]['BustProbabilityIfHit'],
    'BustProbIfHit_pred': y_pred[:sample_size, 1],
    'ImproveIfHit_actual': y_test.iloc[:sample_size]['ImproveHandWithoutBustingIfHit'],
    'ImproveIfHit_pred': y_pred[:sample_size, 2],
    'DealerBeatsIfStand_actual': y_test.iloc[:sample_size]['IfStandOddsDealersSecondCardMakesThemBeatUs'],
    'DealerBeatsIfStand_pred': y_pred[:sample_size, 3],
})
print(comparison_df)

joblib.dump(model, "multi_output_model.joblib")