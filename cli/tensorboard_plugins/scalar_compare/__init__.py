from pathlib import Path

from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import scalars_plugin
from werkzeug.exceptions import NotFound
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.wrappers import Response


class ScalarComparePlugin(scalars_plugin.ScalarsPlugin):
    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path="/ui/scalar_compare/entry.js",
            tab_name="Scalar Compare",
            remove_dom=True,
        )

    def get_plugin_apps(self):
        apps = super().get_plugin_apps()
        root = Path(__file__).resolve().parent.parent
        assets = (
            "scalar_compare/entry.js",
            "scalar_compare/data.js",
            "scalar_compare/style.css",
            "shared/histogram/vendor/d3-esm.js",
        )
        static_app = SharedDataMiddleware(
            NotFound(),
            {f"/data/plugin/{self.plugin_name}/ui/{asset}": str(root / asset) for asset in assets},
            cache_timeout=0,
        )

        def serve_static(environ, start_response):
            response = Response.from_app(static_app, environ)
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response(environ, start_response)

        apps["/ui/*"] = serve_static
        return apps
