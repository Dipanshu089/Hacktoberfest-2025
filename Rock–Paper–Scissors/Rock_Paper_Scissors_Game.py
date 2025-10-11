import random

def get_computer_choice():
    return random.choice(["rock", "paper", "scissors"])

def get_winner(player, computer):
    if player == computer:
        return "tie"
    elif (player == "rock" and computer == "scissors") or \
         (player == "paper" and computer == "rock") or \
         (player == "scissors" and computer == "paper"):
        return "player"
    else:
        return "computer"

def main():
    print("=== Rock, Paper, Scissors ===")
    print("Type 'rock', 'paper', or 'scissors'")
    print("Type 'q' or 'quit' to exit\n")

    player_score = 0
    computer_score = 0
    ties = 0

    while True:
        player = input("Your move > ").lower().strip()
        if player in ["q", "quit", "exit"]:
            print("\nThanks for playing!")
            break
        if player not in ["rock", "paper", "scissors"]:
            print("Invalid choice. Try again.")
            continue

        computer = get_computer_choice()
        print(f"You chose: {player}")
        print(f"Computer chose: {computer}")

        result = get_winner(player, computer)

        if result == "tie":
            print("Result: It's a tie!\n")
            ties += 1
        elif result == "player":
            print("Result: You win! 🎉\n")
            player_score += 1
        else:
            print("Result: Computer wins 😢\n")
            computer_score += 1

        print(f"Score -> You: {player_score} | Computer: {computer_score} | Ties: {ties}\n")

    print("=== Final Score ===")
    print(f"You: {player_score} | Computer: {computer_score} | Ties: {ties}")

if __name__ == "__main__":
    main()
