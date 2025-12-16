"""
Unit tests for spin_symmetric_configs function from 08_spin_recovery.py
"""
import pytest
import sys
from pathlib import Path

# Add research directory to path to import the function
research_dir = Path(__file__).parent.parent / "research"
sys.path.insert(0, str(research_dir))

# Import directly from the module file (with underscore prefix removed for import)
import importlib.util
spec = importlib.util.spec_from_file_location("spin_recovery", research_dir / "08_spin_recovery" / "08_spin_recovery.py")
spin_recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spin_recovery)
spin_symmetric_configs = spin_recovery.spin_symmetric_configs


class TestSpinSymmetricConfigs:
    """Test suite for spin_symmetric_configs function."""

    def test_all_closed_shell(self):
        """Test with all orbitals doubly occupied (closed shell)."""
        # "1111" = alpha=[1,1], beta=[1,1] -> all doubly occupied
        result = spin_symmetric_configs("1111")
        assert len(result) == 1
        assert result[0] == "1111"

    def test_all_empty(self):
        """Test with all orbitals empty."""
        # "0000" = alpha=[0,0], beta=[0,0] -> all empty
        result = spin_symmetric_configs("0000")
        assert len(result) == 1
        assert result[0] == "0000"

    def test_single_alpha_electron(self):
        """Test with a single alpha electron."""
        # "1000" = alpha=[1,0], beta=[0,0] -> one alpha in first orbital
        result = spin_symmetric_configs("1000")
        assert len(result) == 1
        assert result[0] == "1000"

    def test_single_beta_electron(self):
        """Test with a single beta electron."""
        # "0010" = alpha=[0,0], beta=[1,0] -> one beta in first orbital
        result = spin_symmetric_configs("0010")
        assert len(result) == 1
        assert result[0] == "0010"

    def test_two_open_shell_alpha(self):
        """Test with two alpha electrons in different orbitals."""
        # "1100" = alpha=[1,1], beta=[0,0] -> two alpha electrons
        result = spin_symmetric_configs("1100")
        # Should have only 1 configuration (both alpha)
        assert len(result) == 1
        assert "1100" in result

    def test_one_alpha_one_beta_same_orbital(self):
        """Test with one doubly occupied orbital."""
        # "1010" = alpha=[1,0], beta=[1,0] -> first orbital doubly occupied
        result = spin_symmetric_configs("1010")
        assert len(result) == 1
        assert result[0] == "1010"

    def test_mixed_open_shells(self):
        """Test with mixed alpha and beta open shells."""
        # "10100101" = alpha=[1,0,1,0], beta=[0,1,0,1]
        # Orbital 0: alpha only (spin=1)
        # Orbital 1: beta only (spin=-1)
        # Orbital 2: alpha only (spin=1)
        # Orbital 3: beta only (spin=-1)
        # Open shells: [1, -1, 1, -1]
        # Permutations of [1, -1, 1, -1] are limited
        result = spin_symmetric_configs("10100101")
        assert len(result) > 0
        assert "10100101" in result
        # Check all results have same number of electrons
        for config in result:
            assert config.count('1') == "10100101".count('1')

    def test_two_electrons_different_spins(self):
        """Test with one alpha and one beta in different orbitals."""
        # "100001" = alpha=[1,0,0], beta=[0,0,1]
        # Orbital 0: alpha only
        # Orbital 1: empty
        # Orbital 2: beta only
        # Open shells: [1, 0, -1]
        # Permutations: [1, 0, -1] and [-1, 0, 1]
        result = spin_symmetric_configs("100001")
        assert len(result) == 2
        assert "100001" in result
        assert "001100" in result

    def test_three_orbitals_mixed(self):
        """Test with three orbitals: one doubly occupied, two open shell."""
        # "110010" = alpha=[1,1,0], beta=[0,1,0]
        # Orbital 0: alpha only (open, spin=1)
        # Orbital 1: doubly occupied (closed, spin=0)
        # Orbital 2: empty (closed, spin=0)
        # Only one open shell, so only one configuration
        result = spin_symmetric_configs("110010")
        assert len(result) == 1
        assert result[0] == "110010"

    def test_four_orbitals_all_singly_occupied(self):
        """Test with four orbitals all singly occupied with alpha."""
        # "11110000" = alpha=[1,1,1,1], beta=[0,0,0,0]
        # All open shells with alpha spin
        result = spin_symmetric_configs("11110000")
        assert len(result) == 1
        assert result[0] == "11110000"

    def test_two_alpha_two_beta_different_orbitals(self):
        """Test with two alphas and two betas in different orbitals."""
        # "11000011" = alpha=[1,1,0,0], beta=[0,0,1,1]
        # Orbital 0: alpha (spin=1)
        # Orbital 1: alpha (spin=1)
        # Orbital 2: beta (spin=-1)
        # Orbital 3: beta (spin=-1)
        # Permutations of [1, 1, -1, -1]
        result = spin_symmetric_configs("11000011")
        assert len(result) > 1
        # Should include various arrangements
        assert "11000011" in result
        # Check that all have 2 alphas and 2 betas
        for config in result:
            alpha_part = config[:4]
            beta_part = config[4:]
            assert alpha_part.count('1') + beta_part.count('1') == 4

    def test_symmetry_preservation(self):
        """Test that electron count is preserved across all configurations."""
        config = "10110010"
        result = spin_symmetric_configs(config)
        original_count = config.count('1')
        for cfg in result:
            assert cfg.count('1') == original_count

    def test_invalid_odd_length(self):
        """Test that odd-length bitstring raises ValueError."""
        with pytest.raises(ValueError, match="Bitstring must have even length"):
            spin_symmetric_configs("101")

    def test_empty_string(self):
        """Test with empty string."""
        result = spin_symmetric_configs("")
        assert len(result) == 1
        assert result[0] == ""

    def test_minimal_two_bits(self):
        """Test with minimal valid input (2 bits)."""
        # "10" = alpha=[1], beta=[0]
        result = spin_symmetric_configs("10")
        assert len(result) == 1
        assert result[0] == "10"

    def test_unique_configurations(self):
        """Test that all returned configurations are unique."""
        config = "11000011"
        result = spin_symmetric_configs(config)
        assert len(result) == len(set(result))

    def test_all_results_valid_bitstrings(self):
        """Test that all results are valid bitstrings."""
        config = "10100101"
        result = spin_symmetric_configs(config)
        for cfg in result:
            assert all(c in '01' for c in cfg)
            assert len(cfg) == len(config)

    def test_complex_configuration(self):
        """Test with a more complex configuration."""
        # "11011010" = alpha=[1,1,0,1], beta=[1,0,1,0]
        # Orbital 0: doubly occupied (closed)
        # Orbital 1: alpha only (open, spin=1)
        # Orbital 2: beta only (open, spin=-1)
        # Orbital 3: alpha only (open, spin=1)
        # Open shells: [1, -1, 1]
        result = spin_symmetric_configs("11011010")
        assert len(result) > 0
        # All should maintain the same total occupation
        for cfg in result:
            assert cfg.count('1') == "11011010".count('1')
        # Should include the original
        assert "11011010" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

