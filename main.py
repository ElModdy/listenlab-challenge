import json
import httpx
import numpy as np
import time
import sys
import random
import backoff
import os
from policy import optimize_acceptance_policy

CLIENT_ID = os.getenv("CLIENT_ID")

BEST_REJECTED = [790, 3314, 4317]

mu_array = [
    {(0, 0): np.float64(0.499), (0, 1): np.float64(0.17850000000000002), (1, 0): np.float64(0.1785), (1, 1): np.float64(0.14400000000000002)},
    {(0, 0, 0, 0): np.float64(0.03425090875645645), (0, 0, 0, 1): np.float64(0.04712353132226759), (0, 0, 1, 0): np.float64(0.0014340346430559681), (0, 0, 1, 1): np.float64(0.0017665252782197165), (0, 1, 0, 0): np.float64(0.030651434603756727), (0, 1, 0, 1): np.float64(0.24927812531751903), (0, 1, 1, 0): np.float64(0.003433621996730672), (0, 1, 1, 1): np.float64(0.005561818081993641), (1, 0, 0, 0): np.float64(0.4193666991917123), (1, 0, 0, 1): np.float64(0.013378360729563535), (1, 0, 1, 0): np.float64(0.003848357408775115), (1, 0, 1, 1): np.float64(0.008831582669949203), (1, 1, 0, 0): np.float64(0.09733195744807445), (1, 1, 0, 1): np.float64(0.04634898263064985), (1, 1, 1, 0): np.float64(0.011682985951438242), (1, 1, 1, 1): np.float64(0.02571107396983744)},
    {(0, 0, 0, 0, 0, 0): np.float64(0.014126149197971043), (0, 0, 0, 0, 0, 1): np.float64(0.022117704521008967), (0, 0, 0, 0, 1, 0): np.float64(0.00034634898771087973), (0, 0, 0, 0, 1, 1): np.float64(0.000547629173292945), (0, 0, 0, 1, 0, 0): np.float64(0.0001169359239140866), (0, 0, 0, 1, 0, 1): np.float64(0.00022657375215248777), (0, 0, 0, 1, 1, 0): np.float64(8.35946273900416e-05), (0, 0, 0, 1, 1, 1): np.float64(0.0002786230928043914), (0, 0, 1, 0, 0, 0): np.float64(0.012197606777570655), (0, 0, 1, 0, 0, 1): np.float64(0.06456541562144338), (0, 0, 1, 0, 1, 0): np.float64(0.00015215484503282558), (0, 0, 1, 0, 1, 1): np.float64(0.0003494038734082479), (0, 0, 1, 1, 0, 0): np.float64(0.00105219320153558), (0, 0, 1, 1, 0, 1): np.float64(0.0009084031056606927), (0, 0, 1, 1, 1, 0): np.float64(0.00046823018687584334), (0, 0, 1, 1, 1, 1): np.float64(0.0004380331122278951), (0, 1, 0, 0, 0, 0): np.float64(0.013576862901512063), (0, 1, 0, 0, 0, 1): np.float64(0.007800688183824272), (0, 1, 0, 0, 1, 0): np.float64(0.00011623432835302858), (0, 1, 0, 0, 1, 1): np.float64(0.0008404231063097845), (0, 1, 0, 1, 0, 0): np.float64(0.00019420607538124437), (0, 1, 0, 1, 0, 1): np.float64(0.0009707150754113511), (0, 1, 0, 1, 1, 0): np.float64(0.0009767236305967847), (0, 1, 0, 1, 1, 1): np.float64(0.00013058742236647013), (0, 1, 1, 0, 0, 0): np.float64(0.15290876384089142), (0, 1, 1, 0, 0, 1): np.float64(0.01862920692060476), (0, 1, 1, 0, 1, 0): np.float64(0.0006866733752226067), (0, 1, 1, 0, 1, 1): np.float64(0.00039273434584315327), (0, 1, 1, 1, 0, 0): np.float64(0.0025218401523202922), (0, 1, 1, 1, 0, 1): np.float64(0.0012657347487977431), (0, 1, 1, 1, 1, 0): np.float64(0.0006754819477215635), (0, 1, 1, 1, 1, 1): np.float64(0.0008381239448435452), (1, 0, 0, 0, 0, 0): np.float64(0.006936330119858068), (1, 0, 0, 0, 0, 1): np.float64(0.15765808569562212), (1, 0, 0, 0, 1, 0): np.float64(0.0012984789604196393), (1, 0, 0, 0, 1, 1): np.float64(0.006302409000349008), (1, 0, 0, 1, 0, 0): np.float64(0.0005398229960514344), (1, 0, 0, 1, 0, 1): np.float64(0.0022208795525560515), (1, 0, 0, 1, 1, 0): np.float64(0.001007773798057936), (1, 0, 0, 1, 1, 1): np.float64(0.0038426606008407356), (1, 0, 1, 0, 0, 0): np.float64(0.013369296045674548), (1, 0, 1, 0, 0, 1): np.float64(0.10416094727543128), (1, 0, 1, 0, 1, 0): np.float64(0.000359832662328876), (1, 0, 1, 0, 1, 1): np.float64(0.002714207242877456), (1, 0, 1, 1, 0, 0): np.float64(0.0012400664610884423), (1, 0, 1, 1, 0, 1): np.float64(0.0029215897524611074), (1, 0, 1, 1, 1, 0): np.float64(0.0018151852085199604), (1, 0, 1, 1, 1, 1): np.float64(0.002137434627863321), (1, 1, 0, 0, 0, 0): np.float64(0.04386413610785881), (1, 1, 0, 0, 0, 1): np.float64(0.013054703174752099), (1, 1, 0, 0, 1, 0): np.float64(0.0019332141841306404), (1, 1, 0, 0, 1, 1): np.float64(0.003888602357026528), (1, 1, 0, 1, 0, 0): np.float64(0.0011100972277312653), (1, 1, 0, 1, 0, 1): np.float64(0.0002141094943945292), (1, 1, 0, 1, 1, 0): np.float64(0.0006730909330628256), (1, 1, 0, 1, 1, 1): np.float64(0.002005605797288365), (1, 1, 1, 0, 0, 0): np.float64(0.2612416897510924), (1, 1, 1, 0, 0, 1): np.float64(0.025936653864884092), (1, 1, 1, 0, 1, 0): np.float64(0.0003222279143723704), (1, 1, 1, 0, 1, 1): np.float64(0.0014651856433220128), (1, 1, 1, 1, 0, 0): np.float64(0.004556003219548525), (1, 1, 1, 1, 0, 1): np.float64(0.0032565892609951706), (1, 1, 1, 1, 1, 0): np.float64(0.00303275441020418), (1, 1, 1, 1, 1, 1): np.float64(0.004420336659336145)}
]

attr_to_index_array = [
    {'well_dressed': 0, 'young': 1},
    {'techno_lover': 0, 'well_connected': 1, 'creative': 2, 'berlin_local': 3},
    {'underground_veteran': 0, 'international': 1, 'fashion_forward': 2, 'queer_friendly': 3, 'vinyl_collector': 4, 'german_speaker': 5}
]

def constraints_to_counts(constraint_items, attribute_index_map):
    """Transform constraint JSON list to numpy array of remaining required counts.

    constraint_items : list[dict] entries contain 'attribute' and 'minCount'.
    attribute_index_map : dict[str,int] maps attribute name to column index.
    """
    counts = np.zeros(len(attribute_index_map), dtype=int)
    for item in constraint_items:
        name = item.get("attribute")
        if name in attribute_index_map:
            counts[attribute_index_map[name]] = item.get("minCount", 0)
    return counts

def person_to_state(person_obj, attribute_index_map):
    """Convert API person payload into (index, state_tuple, raw_attributes).

    person_obj : dict with personIndex, attributes.
    attribute_index_map : dict[str,int].
    """
    person_index = person_obj["personIndex"]
    attributes = person_obj["attributes"]
    key = [0] * len(attribute_index_map)
    for attr, j in attribute_index_map.items():
        key[j] = int(attributes.get(attr, 0))
    return person_index, tuple(key), attributes

# ----------------- Client -----------------
class GameClient:
    """Encapsulates API access and game execution for a single scenario."""

    def __init__(self, api_base, player_id, scenario):
        """Create client.

        api_base : str base URL
        player_id : str identifier
        scenario : int scenario number (1..n)
        """
        self.api_base = api_base
        self.player_id = player_id
        self.scenario = scenario
        self.client = httpx.Client(
            timeout=30.0,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            headers={
                "User-Agent": "Mozilla/5.0 ...",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            },
            http2=False,
        )

    def close(self):
        """Close underlying HTTP client."""
        self.client.close()

    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.TimeoutException),
        max_tries=5,
        jitter=None
    )
    @backoff.on_predicate(
        backoff.expo,
        max_tries=5,
        predicate=lambda r: r is not None and r.status_code >= 500
    )
    def _get(self, url):
        """Issue GET request (retry backoff applied by decorators)."""
        return self.client.get(url)

    def new_game(self):
        """Start a new game session and return its JSON payload."""
        url = f"{self.api_base}/new-game?scenario={self.scenario}&playerId={self.player_id}"
        attempts = 0
        while attempts < 200:
            resp = self._get(url)
            if resp.status_code == 429:
                attempts += 1
                time.sleep(5)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Failed to start game after 200 retries (HTTP 429)")

    def decide(self, game_id, person_index, accept=None):
        """Submit decision for a person and return next state JSON."""
        url = f"{self.api_base}/decide-and-next?gameId={game_id}&personIndex={person_index}"
        if accept is not None:
            url += f"&accept={'true' if accept else 'false'}"
        resp = self._get(url)
        resp.raise_for_status()
        return resp.json()

    def play(self):
        """Run the main game loop until win, abort, or quota exhaustion.

        Returns dict with success (bool), rejected (int), aborted (bool).
        """
        state_distribution = mu_array[self.scenario - 1]
        attribute_index_map = attr_to_index_array[self.scenario - 1]
        game_payload = self.new_game()
        game_id = game_payload["gameId"]
        remaining_required_counts = constraints_to_counts(game_payload["constraints"], attribute_index_map)
        remaining_acceptances = 1000
        rejected_count = 0
        state = self.decide(game_id, 0)
        policy_cache = None
        recompute_needed = False
        aborted = False
        while state.get("status") == "running" and remaining_acceptances > 0:
            person_index, state_key, _ = person_to_state(state["nextPerson"], attribute_index_map)
            if policy_cache is None or recompute_needed:
                policy_cache = optimize_acceptance_policy(
                    remaining_acceptances,
                    remaining_required_counts,
                    state_distribution,
                )["policy"]
                recompute_needed = False
            accept = np.random.rand() < policy_cache.get(state_key, 0)
            if accept:
                recompute_needed = True
                remaining_acceptances -= 1
                for i, bit in enumerate(state_key):
                    if bit == 1:
                        remaining_required_counts[i] -= 1
            else:
                rejected_count += 1
                if rejected_count > BEST_REJECTED[self.scenario - 1]:
                    aborted = True
                    break
            state = self.decide(game_id, person_index, accept)
        if not aborted and rejected_count < BEST_REJECTED[self.scenario - 1]:
            BEST_REJECTED[self.scenario - 1] = rejected_count
        return {
            "success": state.get("status") == "won" if not aborted else False,
            "rejected": rejected_count,
            "aborted": aborted,
        }

def main():
    """CLI entry: run one game for the requested scenario."""
    scenario = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    time.sleep(random.random() * 1.0)
    client = GameClient("https://berghain.challenges.listenlabs.ai", CLIENT_ID, scenario)
    try:
        result = client.play()
        print(f"Game: {result}")
    finally:
        client.close()

if __name__ == "__main__":
    main()

