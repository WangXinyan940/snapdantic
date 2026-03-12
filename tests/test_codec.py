"""Tests for codec.py (TypeCodec + CodecRegistry)."""
import pytest

from snapdantic.codec import CodecRegistry, TypeCodec


# ── Fixture codecs ────────────────────────────────────────────────────────────

class Animal:
    def __init__(self, name: str):
        self.name = name


class Dog(Animal):
    pass


class _AnimalCodec(TypeCodec):
    @property
    def target_type(self):
        return Animal

    def encode(self, obj, db, zarr):
        return {"name": obj.name}

    def decode(self, refs, db, zarr):
        return Animal(refs["name"])


class TestCodecRegistry:
    def setup_method(self):
        # Clear the registry between tests to avoid cross-test pollution
        CodecRegistry._registry.clear()

    def test_register_and_get_by_key(self):
        codec = _AnimalCodec()
        CodecRegistry.register(codec)
        key = CodecRegistry._make_key(Animal)
        found = CodecRegistry.get_by_key(key)
        assert found is codec

    def test_get_for_value_direct_match(self):
        CodecRegistry.register(_AnimalCodec())
        result = CodecRegistry.get_for_value(Animal("rex"))
        assert result is not None
        key, codec = result
        assert isinstance(codec, _AnimalCodec)

    def test_get_for_value_mro_fallback(self):
        """Subclass (Dog) should inherit parent codec (Animal)."""
        CodecRegistry.register(_AnimalCodec())
        result = CodecRegistry.get_for_value(Dog("rex"))
        assert result is not None
        key, codec = result
        assert isinstance(codec, _AnimalCodec)

    def test_get_for_value_no_match_returns_none(self):
        result = CodecRegistry.get_for_value(42)
        assert result is None

    def test_codec_decorator(self):
        # Define target type outside the codec class to avoid closure issue
        class Cat:
            pass

        @CodecRegistry.codec
        class CatCodec(TypeCodec):
            @property
            def target_type(self):
                return Cat

            def encode(self, obj, db, zarr):
                return {}

            def decode(self, refs, db, zarr):
                return Cat()

        key = CodecRegistry._make_key(Cat)
        assert key in CodecRegistry._registry

    def test_collect_uuids_default(self):
        codec = _AnimalCodec()
        refs = {"a": "uuid-1", "b": None, "c": "uuid-2"}
        uuids = codec.collect_uuids(refs)
        assert uuids == {"uuid-1", "uuid-2"}

    def test_make_key_format(self):
        key = CodecRegistry._make_key(Animal)
        assert "." in key
        assert "Animal" in key

    def test_register_overwrite(self):
        """Registering a new codec for the same type should overwrite."""
        class _AnimalCodec2(_AnimalCodec):
            pass

        c1 = _AnimalCodec()
        c2 = _AnimalCodec()
        CodecRegistry.register(c1)
        CodecRegistry.register(c2)
        key = CodecRegistry._make_key(Animal)
        assert CodecRegistry._registry[key] is c2
