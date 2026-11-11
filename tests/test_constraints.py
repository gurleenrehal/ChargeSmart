import numpy as np
import pytest

from src.ev_simulator import ChargingSim, ScenarioGenerator, SimConfig


@pytest.mark.parametrize("seed", range(10))
def test_physical_constraints_hold_under_adversarial_requests(seed):
    cfg = SimConfig(grid_capacity_kw=9.0)
    sc = ScenarioGenerator(cfg).generate(seed)
    sim = ChargingSim(cfg, sc)
    rng = np.random.default_rng(seed)
    while not sim.done:
        t = sim.t
        occ = sim.occupant().copy()
        info = sim.step(rng.uniform(0, 50, sim.P))             # absurd requests
        p = info["port_power"]
        assert p.sum() <= cfg.grid_capacity_kw + 1e-9           # grid capacity
        assert (p[occ < 0] == 0).all()                          # nothing into empty ports / after departure
        for port in np.flatnonzero(occ >= 0):
            assert p[port] <= sc.pmax_kw[occ[port]] + 1e-9      # charger / EV limit
    assert (sim.soc <= sc.soc_target + 1e-9).all()              # never above target
    assert (sim.ev_power.sum(axis=0) <= cfg.grid_capacity_kw + 1e-9).all()


def test_no_charging_after_departure():
    cfg = SimConfig()
    sc = ScenarioGenerator(cfg).generate(4)
    sim = ChargingSim(cfg, sc)
    while not sim.done:
        sim.step(np.full(sim.P, 100.0))
    for i in range(sc.n_evs):
        assert sim.ev_power[i, sc.departure[i]:].sum() == 0
        assert sim.ev_power[i, :sc.arrival[i]].sum() == 0


def test_inactive_ports_are_never_used():
    cfg = SimConfig(n_stations=1, ports_per_station=1)
    sc = ScenarioGenerator(cfg).generate(1)
    assert set(sc.port.tolist()) <= {0}


def test_energy_balance():
    cfg = SimConfig()
    sc = ScenarioGenerator(cfg).generate(7)
    sim = ChargingSim(cfg, sc)
    while not sim.done:
        sim.step(np.full(sim.P, 7.2))
    gained = ((sim.soc - sc.soc0) * sc.capacity_kwh).sum()
    assert np.isclose(gained, sim.delivered.sum())
    assert np.isclose(sim.delivered.sum(), sim.load.sum() * cfg.dt_hours)
