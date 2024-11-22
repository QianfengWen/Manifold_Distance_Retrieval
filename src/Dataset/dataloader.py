from abc import ABC, abstractmethod

class Dataloader(ABC):    
    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_questions(self):
        pass

    @abstractmethod
    def load_passages(self):
        pass

    @abstractmethod
    def create_relevance_map(self):
        pass


    def load_data(self):
        self.dataset = self.load_dataset()
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        relevance_map = self.create_relevance_map()
        
        return question_ids, question_texts, passage_ids, passage_texts, relevance_map