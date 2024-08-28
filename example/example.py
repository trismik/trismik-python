from typing import List

from dotenv import load_dotenv

from trismik import Trismik, TrismikMultipleChoiceTextItem, TrismikResult

load_dotenv() # Load environment variables from .env file if present

class TrismikExample:
    def __init__(self):
        self.trismik = Trismik()
        self.token = self.authenticate()

    def authenticate(self) -> str:
        """Authenticate with the Trismik API and return the token."""
        return self.trismik.authenticate().token

    def run_test(self, session_url: str) -> List[TrismikResult]:
        """Run the test session and return the results."""
        item_counter = 0
        item = self.trismik.current_item(session_url, self.token)

        while item is not None:
            if isinstance(item, TrismikMultipleChoiceTextItem):
                print(f"Responding to item #{item_counter}: {item.question[:20]}...")
                choice_id = self.pick_first_choice_id(item)
                item = self.trismik.respond_to_current_item(session_url, choice_id, self.token)
                item_counter += 1
            else:
                raise RuntimeError("Encountered unknown item type")

        return self.trismik.results(session_url, self.token)

    @staticmethod
    def pick_first_choice_id(item: TrismikMultipleChoiceTextItem) -> str:
        """Pick the first choice for a multiple choice text item."""
        return item.choices[0].id

    def run_first_available_test(self) -> List[TrismikResult]:
        """Run the first available test and return the results."""
        available_tests = self.trismik.available_tests(self.token)
        if not available_tests:
            raise RuntimeError("No available tests found")

        session = self.trismik.create_session(available_tests[0].id, self.token)
        return self.run_test(session.url)

def main():
    """Main function to run the first available test and print the results."""
    example = TrismikExample()
    results = example.run_first_available_test()

    for result in results:
        print(f"{result.trait} ({result.name}): {result.value}")

if __name__ == "__main__":
    main()
