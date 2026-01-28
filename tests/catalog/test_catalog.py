from altimetry.io import AltimetryCatalog


def test_catalog_init():
    catalog = AltimetryCatalog.load()

    assert catalog

    # TODO: Replace with real tests
    catalog.show_dataset(dtype="cls_table")
    catalog.show_dataset(dtype="sc_collection")
    catalog.show_dataset(dtype="file_collection")
    catalog.show_dataset(dtype=["cls_table", "file_collection"])

    catalog.show_dataset(dtype=["cls_table", "file_collection"], containing="jason")

    catalog.show_dataset(containing="jason")
    catalog.show_dataset(containing="something_not_existing")
