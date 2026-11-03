from multimodal_retrieval.data import _stable_rank, is_allowed_license_url


def test_license_filter_allows_only_attribution_and_cc0_metadata() -> None:
    assert is_allowed_license_url("http://creativecommons.org/licenses/by/2.0/")
    assert is_allowed_license_url("https://creativecommons.org/publicdomain/zero/1.0")
    assert not is_allowed_license_url("http://creativecommons.org/licenses/by-sa/2.0/")
    assert not is_allowed_license_url("http://creativecommons.org/licenses/by-nc/2.0/")
    assert not is_allowed_license_url("http://creativecommons.org/licenses/by-nd/2.0/")
    assert not is_allowed_license_url("http://flickr.com/commons/usage/")


def test_sample_rank_is_deterministic_and_id_specific() -> None:
    assert _stable_rank(100) == _stable_rank(100)
    assert _stable_rank(100) != _stable_rank(101)
