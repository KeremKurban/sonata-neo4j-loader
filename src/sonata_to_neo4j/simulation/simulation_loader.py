# Bluepysnap Simulation loader.
import logging
from typing import Any, Dict

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SimulationLoader(BaseModel):
    """
    A class to load and process simulation results from CoreNeuron.

    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary for the simulation.

    Methods
    -------
    load_results():
        Loads simulation results from CoreNeuron.
    process_results():
        Processes the loaded simulation results.
    """

    config: Dict[str, Any]

    def load_results(self):
        """
        Loads simulation results from CoreNeuron.

        This method should contain the logic to interface with CoreNeuron
        and retrieve the simulation results for further processing.
        """
        pass

    def process_results(self):
        """
        Processes the loaded simulation results.

        This method should contain the logic to process the simulation
        results that have been loaded, such as data analysis or transformation.
        """
        pass
