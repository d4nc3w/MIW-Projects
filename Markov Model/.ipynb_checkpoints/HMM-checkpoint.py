import numpy as np

#--------------------------------MODEL------------------------------------
class HiddenMarkovModel:
    def __init__(self, states: list[str]) -> None:
        self.states = states
        self.transitions = {}

    def predict_next_state(self, current_state: str) -> str:
        if current_state not in self.transitions or not self.transitions[current_state]:
            return np.random.choice(self.states)

        next_states = list(self.transitions[current_state].keys())
        counts = list(self.transitions[current_state].values())

        return np.random.choice(next_states, p=np.array(counts) / sum(counts))

    def learn_transition(self, state_from: str, state_to: str) -> None:
        if state_from not in self.transitions:
            self.transitions[state_from] = {}

        if state_to not in self.transitions[state_from]:
            self.transitions[state_from][state_to] = 0

        self.transitions[state_from][state_to] += 1

    def generate_sentence(self, length: int) -> str:
        start_word = np.random.choice(self.states).capitalize()
        sentence = [start_word]

        for i in range(length - 1):
            next_word = self.predict_next_state(sentence[-1])
            sentence.append(next_word.lower())
        return " ".join(sentence)

#--------------------------------DATA------------------------------------
def read_data(filename: str) -> list[str]:
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().replace(".\n", " ").split()
    return words

data = read_data("yodaQuotes.txt")

#--------------------------------TESTING----------------------------------
model = HiddenMarkovModel(list(set(data)))
for i in range(len(data) - 1):
    model.learn_transition(data[i], data[i + 1])

print(model.generate_sentence(8))
print(model.generate_sentence(8))
print(model.generate_sentence(8))
print(model.generate_sentence(8))
print(model.generate_sentence(8))
print(model.generate_sentence(8))
print(model.generate_sentence(8))
print(model.generate_sentence(8))

