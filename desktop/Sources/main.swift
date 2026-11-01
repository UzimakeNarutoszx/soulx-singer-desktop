import AppKit
import Foundation
import WebKit

private enum DesktopConfig {
    static let project = URL(fileURLWithPath: ProcessInfo.processInfo.environment["SOULX_PROJECT_ROOT"] ?? FileManager.default.currentDirectoryPath, isDirectory: true)
    static let python = project.appendingPathComponent(".venv/bin/python")
    static let entrypoint = project.appendingPathComponent("webui.py")
    static let serviceURL = URL(string: "http://127.0.0.1:7860")!
    static let configURL = serviceURL.appendingPathComponent("config")
    static let expectedTitle = "SoulX-Singer"
    static let timeout: TimeInterval = 180
    static let logURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/SoulX-Singer/desktop.log")
}

private final class ServiceController {
    private let queue = DispatchQueue(label: "com.soulailab.soulxsinger.service")
    private let stateLock = NSLock()
    private var process: Process?
    private var managedPID: pid_t?
    private var stopping = false

    var onProgress: ((String) -> Void)?
    var onReady: (() -> Void)?
    var onFailure: ((String) -> Void)?

    func start() {
        queue.async { [weak self] in
            guard let self else { return }
            self.stateLock.withLock { self.stopping = false }
            do {
                try self.validateInstallation()
                self.progress("正在检查本地服务…")

                if let listener = self.listenerPID() {
                    guard self.isProjectService(pid: listener), self.hasExpectedConfig() else {
                        throw DesktopError.message("端口 7860 已被其他程序占用。SoulX-Singer 未启动，也不会结束该程序。")
                    }
                    self.stateLock.withLock { self.managedPID = listener }
                    self.appendLog("Attached to existing project service (PID \(listener)).")
                    self.ready()
                    return
                }

                try self.launchService()
                try self.waitUntilReady()
                self.ready()
            } catch DesktopError.cancelled {
                return
            } catch {
                self.appendLog("Startup failed: \(error.localizedDescription)")
                self.fail(error.localizedDescription)
            }
        }
    }

    func retry() {
        stop()
        start()
    }

    func stop() {
        let pid = stateLock.withLock { () -> pid_t? in
            stopping = true
            let pid = managedPID
            managedPID = nil
            process = nil
            return pid
        }
        guard let pid, pid > 1 else { return }
        appendLog("Stopping managed service (PID \(pid)).")
        kill(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(6)
        while Date() < deadline, kill(pid, 0) == 0 {
            usleep(100_000)
        }
        if kill(pid, 0) == 0 {
            appendLog("Service did not stop in time; sending SIGKILL to PID \(pid).")
            kill(pid, SIGKILL)
        }
    }

    private func validateInstallation() throws {
        let manager = FileManager.default
        guard manager.fileExists(atPath: DesktopConfig.project.path) else {
            throw DesktopError.message("找不到项目目录：\n\(DesktopConfig.project.path)")
        }
        guard manager.isExecutableFile(atPath: DesktopConfig.python.path) else {
            throw DesktopError.message("找不到项目的 Python 环境：\n\(DesktopConfig.python.path)")
        }
        guard manager.fileExists(atPath: DesktopConfig.entrypoint.path) else {
            throw DesktopError.message("找不到歌声合成入口：\n\(DesktopConfig.entrypoint.path)")
        }

        let requiredModels = [
            "pretrained_models/SoulX-Singer/model.pt",
            "pretrained_models/SoulX-Singer-Preprocess/mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt",
            "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
            "pretrained_models/SoulX-Singer-Preprocess/rosvot/rosvot/model.pt",
            "pretrained_models/SoulX-Singer-Preprocess/rosvot/rwbd/model.pt",
            "pretrained_models/SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pt",
        ]
        for relativePath in requiredModels {
            let url = DesktopConfig.project.appendingPathComponent(relativePath)
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]), (values.fileSize ?? 0) > 0 else {
                throw DesktopError.message("模型文件缺失或为空：\n\(url.path)")
            }
        }
    }

    private func launchService() throws {
        try FileManager.default.createDirectory(
            at: DesktopConfig.logURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: DesktopConfig.logURL.path) {
            FileManager.default.createFile(atPath: DesktopConfig.logURL.path, contents: nil)
        }
        let logHandle = try FileHandle(forWritingTo: DesktopConfig.logURL)
        try logHandle.seekToEnd()

        let child = Process()
        child.executableURL = DesktopConfig.python
        child.arguments = [DesktopConfig.entrypoint.path, "--port", "7860", "--server-name", "127.0.0.1"]
        child.currentDirectoryURL = DesktopConfig.project
        child.standardOutput = logHandle
        child.standardError = logHandle
        child.terminationHandler = { _ in try? logHandle.close() }

        appendLog("Starting local service.")
        try child.run()
        stateLock.withLock {
            process = child
            managedPID = child.processIdentifier
        }
        progress("正在加载模型，首次启动可能需要 1–3 分钟…")
    }

    private func waitUntilReady() throws {
        let deadline = Date().addingTimeInterval(DesktopConfig.timeout)
        while Date() < deadline {
            if stateLock.withLock({ stopping }) { throw DesktopError.cancelled }
            if hasExpectedConfig() { return }
            if let process = stateLock.withLock({ process }), !process.isRunning {
                throw DesktopError.message("SoulX-Singer 服务启动失败。请打开日志查看原因。")
            }
            Thread.sleep(forTimeInterval: 1)
        }
        throw DesktopError.message("模型在 180 秒内未完成加载。请打开日志查看原因后重试。")
    }

    private func hasExpectedConfig() -> Bool {
        let semaphore = DispatchSemaphore(value: 0)
        var matches = false
        var request = URLRequest(url: DesktopConfig.configURL)
        request.timeoutInterval = 2
        URLSession.shared.dataTask(with: request) { data, _, _ in
            defer { semaphore.signal() }
            guard let data,
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  object["title"] as? String == DesktopConfig.expectedTitle else { return }
            matches = true
        }.resume()
        _ = semaphore.wait(timeout: .now() + 3)
        return matches
    }

    private func listenerPID() -> pid_t? {
        let task = Process()
        let pipe = Pipe()
        task.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        task.arguments = ["-nP", "-t", "-iTCP:7860", "-sTCP:LISTEN"]
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        guard (try? task.run()) != nil else { return nil }
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?
            .split(whereSeparator: \ .isNewline)
            .compactMap { pid_t($0) }
            .first
    }

    private func isProjectService(pid: pid_t) -> Bool {
        let command = commandOutput("/bin/ps", ["-p", String(pid), "-o", "command="])
        let cwd = commandOutput("/usr/sbin/lsof", ["-a", "-p", String(pid), "-d", "cwd", "-Fn"])
        let commandMatches = command.contains("webui.py") && command.contains("--port 7860")
        let cwdMatches = cwd.split(separator: "\n").contains("n\(DesktopConfig.project.path)"[...])
        return commandMatches && cwdMatches
    }

    private func commandOutput(_ executable: String, _ arguments: [String]) -> String {
        let task = Process()
        let pipe = Pipe()
        task.executableURL = URL(fileURLWithPath: executable)
        task.arguments = arguments
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        guard (try? task.run()) != nil else { return "" }
        task.waitUntilExit()
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    private func appendLog(_ message: String) {
        let formatter = ISO8601DateFormatter()
        let line = "[\(formatter.string(from: Date()))] \(message)\n"
        try? FileManager.default.createDirectory(
            at: DesktopConfig.logURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: DesktopConfig.logURL.path) {
            FileManager.default.createFile(atPath: DesktopConfig.logURL.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: DesktopConfig.logURL) else { return }
        _ = try? handle.seekToEnd()
        if let data = line.data(using: .utf8) { try? handle.write(contentsOf: data) }
        try? handle.close()
    }

    private func progress(_ message: String) {
        DispatchQueue.main.async { [weak self] in self?.onProgress?(message) }
    }

    private func ready() {
        DispatchQueue.main.async { [weak self] in self?.onReady?() }
    }

    private func fail(_ message: String) {
        DispatchQueue.main.async { [weak self] in self?.onFailure?(message) }
    }
}

private enum DesktopError: LocalizedError {
    case message(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .message(let message): return message
        case .cancelled: return "已取消"
        }
    }
}

private final class WorkspaceViewController: NSViewController, WKNavigationDelegate {
    private let service = ServiceController()
    private let contentView = NSView()
    private let loadingView = NSView()
    private let spinner = NSProgressIndicator()
    private let titleLabel = NSTextField(labelWithString: "SoulX-Singer")
    private let statusLabel = NSTextField(labelWithString: "正在启动…")
    private let retryButton = NSButton(title: "重试", target: nil, action: nil)
    private let logButton = NSButton(title: "打开日志", target: nil, action: nil)
    private var webView: WKWebView?

    override func loadView() {
        view = contentView
        contentView.wantsLayer = true
        contentView.layer?.backgroundColor = NSColor(calibratedWhite: 0.97, alpha: 1).cgColor
        configureLoadingView()

        service.onProgress = { [weak self] message in self?.showLoading(message) }
        service.onReady = { [weak self] in self?.showWorkspace() }
        service.onFailure = { [weak self] message in self?.showError(message) }
    }

    override func viewDidAppear() {
        super.viewDidAppear()
        service.start()
    }

    func stopService() {
        service.stop()
    }

    private func configureLoadingView() {
        loadingView.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(loadingView)

        let icon = NSImageView()
        icon.image = NSApp.applicationIconImage
        icon.imageScaling = .scaleProportionallyUpOrDown
        icon.translatesAutoresizingMaskIntoConstraints = false

        titleLabel.font = .systemFont(ofSize: 28, weight: .semibold)
        titleLabel.textColor = .labelColor
        titleLabel.alignment = .center
        statusLabel.font = .systemFont(ofSize: 14)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.alignment = .center
        statusLabel.maximumNumberOfLines = 3
        statusLabel.lineBreakMode = .byWordWrapping

        spinner.style = .spinning
        spinner.controlSize = .regular
        spinner.startAnimation(nil)
        retryButton.bezelStyle = .rounded
        retryButton.target = self
        retryButton.action = #selector(retry)
        retryButton.isHidden = true
        logButton.bezelStyle = .rounded
        logButton.target = self
        logButton.action = #selector(openLog)
        logButton.isHidden = true

        let buttons = NSStackView(views: [retryButton, logButton])
        buttons.orientation = .horizontal
        buttons.spacing = 10
        buttons.alignment = .centerY
        let stack = NSStackView(views: [icon, titleLabel, statusLabel, spinner, buttons])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 14
        stack.setCustomSpacing(20, after: icon)
        stack.translatesAutoresizingMaskIntoConstraints = false
        loadingView.addSubview(stack)

        NSLayoutConstraint.activate([
            loadingView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            loadingView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            loadingView.topAnchor.constraint(equalTo: contentView.topAnchor),
            loadingView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            stack.centerXAnchor.constraint(equalTo: loadingView.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: loadingView.centerYAnchor, constant: -20),
            statusLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 480),
            icon.widthAnchor.constraint(equalToConstant: 112),
            icon.heightAnchor.constraint(equalToConstant: 112),
        ])
    }

    private func showLoading(_ message: String) {
        loadingView.isHidden = false
        statusLabel.stringValue = message
        statusLabel.textColor = .secondaryLabelColor
        spinner.isHidden = false
        spinner.startAnimation(nil)
        retryButton.isHidden = true
        logButton.isHidden = true
    }

    private func showError(_ message: String) {
        loadingView.isHidden = false
        statusLabel.stringValue = message
        statusLabel.textColor = .systemRed
        spinner.stopAnimation(nil)
        spinner.isHidden = true
        retryButton.isHidden = false
        logButton.isHidden = false
    }

    private func showWorkspace() {
        if let webView {
            webView.reload()
            loadingView.isHidden = true
            return
        }
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        let browser = WKWebView(frame: .zero, configuration: configuration)
        browser.navigationDelegate = self
        browser.translatesAutoresizingMaskIntoConstraints = false
        browser.setValue(false, forKey: "drawsBackground")
        contentView.addSubview(browser, positioned: .below, relativeTo: loadingView)
        NSLayoutConstraint.activate([
            browser.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            browser.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            browser.topAnchor.constraint(equalTo: contentView.topAnchor),
            browser.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
        ])
        webView = browser
        browser.load(URLRequest(url: DesktopConfig.serviceURL))
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        loadingView.isHidden = true
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        showError("工作台加载失败：\n\(error.localizedDescription)")
    }

    @objc private func retry() {
        showLoading("正在重新启动…")
        service.retry()
    }

    @objc private func openLog() {
        NSWorkspace.shared.open(DesktopConfig.logURL)
    }
}

private final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var window: NSWindow!
    private let workspace = WorkspaceViewController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "SoulX-Singer"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .visible
        window.minSize = NSSize(width: 1024, height: 700)
        window.contentViewController = workspace
        window.delegate = self
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        NSApp.terminate(nil)
        return false
    }

    func windowWillClose(_ notification: Notification) {
        NSApp.terminate(nil)
    }

    func applicationWillTerminate(_ notification: Notification) {
        workspace.stopService()
    }
}

private let application = NSApplication.shared
private let delegate = AppDelegate()
application.setActivationPolicy(.regular)
application.delegate = delegate
application.run()
