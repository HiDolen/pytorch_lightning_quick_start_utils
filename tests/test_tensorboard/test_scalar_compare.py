import json
import os
import tempfile
import types
import unittest
from unittest import mock

from tensorboard.backend.event_processing import data_provider as event_data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard.plugins.scalar import scalars_plugin
from torch.utils.tensorboard import SummaryWriter
from werkzeug.test import Client
from werkzeug.wrappers import Response

from cli.tensorboard import run_tensorboard
from cli.tensorboard_plugins.scalar_compare import ScalarComparePlugin


class ScalarComparePluginTest(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.TemporaryDirectory(prefix="scalar_compare_test_")
        self.addCleanup(self.logdir.cleanup)
        with SummaryWriter(os.path.join(self.logdir.name, "run")) as writer:
            for step in range(2):
                writer.add_scalar("train/loss", 1.0 - step * 0.25, step)
                writer.add_scalar("val/loss", 1.5 - step * 0.25, step)
        multiplexer = plugin_event_multiplexer.EventMultiplexer()
        multiplexer.AddRunsFromDirectory(self.logdir.name)
        multiplexer.Reload()
        provider = event_data_provider.MultiplexerDataProvider(multiplexer, self.logdir.name)
        self.plugin = ScalarComparePlugin(
            types.SimpleNamespace(data_provider=provider, sampling_hints={"scalars": 20000})
        )
        self.apps = self.plugin.get_plugin_apps()

    def test_native_scalar_routes_are_preserved(self):
        tags = Client(self.apps["/tags"], Response).get("/data/plugin/scalars/tags")
        self.assertEqual(tags.status_code, 200)
        self.assertEqual(set(json.loads(tags.data)["run"]), {"train/loss", "val/loss"})
        client = Client(self.apps["/scalars"], Response)
        for tag, expected in [("train/loss", [1.0, 0.75]), ("val/loss", [1.5, 1.25])]:
            with self.subTest(tag=tag):
                response = client.get(
                    "/data/plugin/scalars/scalars",
                    query_string={"run": "run", "tag": tag, "format": "json"},
                )
                self.assertEqual(response.status_code, 200)
                points = json.loads(response.data)
                self.assertEqual([point[1] for point in points], [0, 1])
                self.assertEqual([point[2] for point in points], expected)

    def test_comparison_tab_uses_scalar_activation(self):
        metadata = self.plugin.frontend_metadata()
        self.assertEqual(metadata.tab_name, "Scalar Compare")
        self.assertEqual(metadata.es_module_path, "/ui/scalar_compare/entry.js")
        self.assertIn("scalars", self.plugin.data_plugin_names())

    def test_cli_registers_comparison_in_place_of_scalars(self):
        with (
            mock.patch("cli.tensorboard.default.get_plugins", return_value=[scalars_plugin.ScalarsPlugin]),
            mock.patch("cli.tensorboard.program.TensorBoard") as constructor,
        ):
            constructor.return_value.main.return_value = 0
            self.assertEqual(run_tensorboard(), 0)
            constructor.assert_called_once_with(plugins=[ScalarComparePlugin])

    def test_static_routes_only_expose_declared_assets(self):
        client = Client(self.apps["/ui/*"], Response)
        for path, content_type in [
            ("scalar_compare/entry.js", "javascript"),
            ("scalar_compare/data.js", "javascript"),
            ("scalar_compare/style.css", "text/css"),
            ("shared/histogram/vendor/d3-esm.js", "javascript"),
        ]:
            with self.subTest(path=path):
                response = client.get("/data/plugin/scalars/ui/" + path)
                self.addCleanup(response.close)
                self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response.content_type)
                self.assertEqual(response.cache_control.max_age, 0)
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        for path in ["scalar_compare/__init__.py", "../plugin_base.py", "unknown.js"]:
            with self.subTest(path=path):
                self.assertEqual(client.get("/data/plugin/scalars/ui/" + path).status_code, 404)


if __name__ == "__main__":
    unittest.main()