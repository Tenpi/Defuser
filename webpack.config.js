const HtmlWebpackPlugin = require("html-webpack-plugin")
const ForkTsCheckerWebpackPlugin = require("fork-ts-checker-webpack-plugin")
const MiniCssExtractPlugin = require("mini-css-extract-plugin")
const TerserJSPlugin = require("terser-webpack-plugin")
const CopyPlugin = require("copy-webpack-plugin")
const nodeExternals = require("webpack-node-externals")
const WebpackObfuscator = require("webpack-obfuscator")
const webpack = require("webpack")
const path = require("path")
let exclude = [/node_modules/, /dist/]
let webExclude = [...exclude, /server.tsx/, /routes/]

const testing = process.env.TESTING === "true"

module.exports = [
  {
    target: "web",
    entry: "./index",
    mode: "production",
    node: {__dirname: false},
    output: {filename: "script.js", chunkFilename: "script.js", path: path.resolve(__dirname, "./dist"), publicPath: ""},
    resolve: {extensions: [".js", ".jsx", ".ts", ".tsx"], alias: {"react-dom$": "react-dom/profiling", "scheduler/tracing": "scheduler/tracing-profiling"},
    fallback: {"process/browser": require.resolve("process/browser"), fs: false, child_process: false, path: require.resolve("path-browserify"), crypto: require.resolve("crypto-browserify"), stream: require.resolve("stream-browserify"), assert: require.resolve("assert/"), zlib: require.resolve("browserify-zlib"), buffer: require.resolve("buffer/"), url: require.resolve("url/"), os: require.resolve("os-browserify/browser")}},
    performance: {hints: false},
    experiments: {asyncWebAssembly: true},
    optimization: {minimize: testing ? false : true, minimizer: [new TerserJSPlugin({extractComments: false})], moduleIds: "named"},
    module: {
      rules: [
        {test: /\.(jpe?g|png|ico|icns|gif|webp|svg|mp3|wav|mp4|webm|ttf|otf|pdf|txt|svg|psd)$/, exclude: webExclude, use: [{loader: "file-loader", options: {name: "[path][name].[ext]"}}]},
        {test: /\.(txt|sql)$/, exclude: webExclude, use: ["raw-loader"]},
        {test: /\.html$/, exclude: webExclude, use: [{loader: "html-loader", options: {minimize: false}}]},
        {test: /\.less$/, exclude: webExclude, use: [{loader: MiniCssExtractPlugin.loader, options: {hmr: true}}, "css-loader", {loader: "less-loader"}]},
        {test: /\.css$/, exclude: webExclude, use: [{loader: MiniCssExtractPlugin.loader}, "css-loader"]},
        {test: /\.(tsx?|jsx?)$/, resolve: {fullySpecified: false}, exclude: webExclude, use: [{loader: "ts-loader", options: {transpileOnly: true}}]},
        {test: /\.m?js$/, resolve: {fullySpecified: false}},
        {test: /\.wasm$/, type: "javascript/auto", loader: "file-loader"},
      ]
    },
    plugins: [
      new ForkTsCheckerWebpackPlugin({typescript: {memoryLimit: 8192}}),
      new webpack.HotModuleReplacementPlugin(),
      new MiniCssExtractPlugin({
        filename: "styles.css",
        chunkFilename: "styles.css"
      }),
      new HtmlWebpackPlugin({
        template: path.resolve(__dirname, "./index.html"),
        minify: false
      }),
      new webpack.ProvidePlugin({
        Buffer: ["buffer", "Buffer"],
      }),
      new webpack.ProvidePlugin({
          process: "process/browser",
      })
    ]
  }
]