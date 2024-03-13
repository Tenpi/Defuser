import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, 
SiteLightnessContext, SocketContext, ThemeContext, ThemeSelectorContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import "./styles/traintag.less"
import axios from "axios"

const ModelConvert: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
    const {themeSelector, setThemeSelector} = useContext(ThemeSelectorContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {socket, setSocket} = useContext(SocketContext)
    const [format, setFormat] = useState("ckpt")
    const [mode, setMode] = useState("single file")
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        let saturation = siteSaturation
        let lightness = siteLightness
        if (themeSelector === "original") {
            if (theme === "light") saturation -= 60
            if (theme === "light") lightness += 90
        } else if (themeSelector === "accessibility") {
            if (theme === "light") saturation -= 90
            if (theme === "light") lightness += 200
            if (theme === "dark") saturation -= 50
            if (theme === "dark") lightness -= 30
        }
        return `hue-rotate(${siteHue - 180}deg) saturate(${saturation}%) brightness(${lightness + 50}%)`
    }

    useEffect(() => {
        const savedFormat = localStorage.getItem("convertFormat")
        if (savedFormat) setFormat(savedFormat)
    }, [])

    useEffect(() => {
        localStorage.setItem("convertFormat", String(format))
    }, [format])

    const convert = async () => {
        const json = {} as any
        json.format = format
        json.mode = mode
        await axios.post("/model-convert", json)
    }

    return (
        <div className="train-tag" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-column">
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Model Input</span>
                    <label htmlFor="img" className="options-bar-img-container" onClick={convert}>
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>
                    </label>
                </div>
                <div className="shade-sketch-box">
                    <div className="shade-sketch-box-row">
                        <button className="shade-sketch-button" style={{backgroundColor: mode === "single file" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setMode("single file")}>single file</button>
                        <button className="shade-sketch-button" style={{backgroundColor: mode === "diffusers" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setMode("diffusers")}>diffusers</button>
                    </div>
                </div>
                <div className="shade-sketch-box">
                    <div className="shade-sketch-box-row">
                        <button className="shade-sketch-button" style={{backgroundColor: format === "ckpt" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setFormat("ckpt")}>ckpt</button>
                        <button className="shade-sketch-button" style={{backgroundColor: format === "safetensors" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setFormat("safetensors")}>safetensors</button>
                        <button className="shade-sketch-button" style={{backgroundColor: format === "pt" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setFormat("pt")}>pt</button>
                        <button className="shade-sketch-button" style={{backgroundColor: format === "bin" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setFormat("bin")}>bin</button>
                    </div>
                </div>
                <span className="train-tag-settings-title">Convert models/finetunes to other formats.</span>
            </div>
        </div>
    )
}

export default ModelConvert