# Bluepysnap Simulaiton loader.
from pydantic import BaseModel

class BaseSimulation(BaseModel):
    config: dict

    def load_data(self) -> None:
        """
        Load data required for the simulation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def process_data(self) -> None:
        """
        Process the loaded data for the simulation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")

