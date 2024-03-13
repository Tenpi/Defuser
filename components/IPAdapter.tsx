import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, ThemeContext, ThemeSelectorContext,
ImageBrightnessContext, ImageContrastContext, ImageHueContext, ImageSaturationContext, IPAdapterContext, IPProcessorContext,
IPWeightContext, IPImageContext, IPAdapterNamesContext, ModelNameContext, IPDrawImageContext, IPMaskImageContext, IPMaskDataContext} from "../Context"
import functions from "../structures/Functions"
import {Dropdown, DropdownButton} from "react-bootstrap"
import Slider from "react-slider"
import radioButtonOff from "../assets/icons/radiobutton-off.png"
import radioButtonOn from "../assets/icons/radiobutton-on.png"
import radioButtonOffLight from "../assets/icons/radiobutton-off-light.png"
import radioButtonOnLight from "../assets/icons/radiobutton-on-light.png"
import checkboxChecked from "../assets/icons/checkbox-checked.png"
import checkbox from "../assets/icons/checkbox.png"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import drawIcon from "../assets/icons/draw.png"
import xIcon from "../assets/icons/x-alt.png"
import fileType from "magic-bytes.js"
import "./styles/controlnet.less"
import axios from "axios"
import path from "path"

const IPAdapter: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
    const {themeSelector, setThemeSelector} = useContext(ThemeSelectorContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {imageHue, setImageHue} = useContext(ImageHueContext)
    const {imageSaturation, setImageSaturation} = useContext(ImageSaturationContext)
    const [imageHover, setImageHover] = useState(false)
    const {ipAdapter, setIPAdapter} = useContext(IPAdapterContext)
    const {ipProcessor, setIPProcessor} = useContext(IPProcessorContext)
    const {ipWeight, setIPWeight} = useContext(IPWeightContext)
    const {ipImage, setIPImage} = useContext(IPImageContext)
    const {ipAdapterNames, setIPAdapterNames} = useContext(IPAdapterNamesContext)
    const {modelName, setModelName} = useContext(ModelNameContext)
    const {ipDrawImage, setIPDrawImage} = useContext(IPDrawImageContext)
    const {ipMaskImage, setIPMaskImage} = useContext(IPMaskImageContext)
    const {ipMaskData, setIPMaskData} = useContext(IPMaskDataContext)
    const [img, setImg] = useState(null) as any
    const [maskImg, setMaskImg] = useState(null) as any 
    const ref = useRef(null) as any
    
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
        const savedIPAdapter = localStorage.getItem("ipAdapter")
        if (savedIPAdapter) setIPAdapter(savedIPAdapter)
        const savedIPProcessor = localStorage.getItem("ipProcessor")
        if (savedIPProcessor) setIPProcessor(savedIPProcessor)
        const savedIPWeight = localStorage.getItem("ipWeight")
        if (savedIPWeight) setIPWeight(Number(savedIPWeight))
        const savedIPImage = localStorage.getItem("ipImage")
        if (savedIPImage) setIPImage(savedIPImage)
    }, [])

    useEffect(() => {
        localStorage.setItem("ipAdapter", String(ipAdapter))
        localStorage.setItem("ipProcessor", String(ipProcessor))
        localStorage.setItem("ipWeight", String(ipWeight))
        localStorage.setItem("ipImage", String(ipImage))
    }, [ipAdapter, ipProcessor, ipWeight, ipImage])

    const getRadioButton = (condition: boolean) => {
        if (theme === "light") {
            return condition ? radioButtonOnLight : radioButtonOffLight
        } else {
            return condition ? radioButtonOn : radioButtonOff
        }
    }

    const loadImage = async (event: any) => {
        const file = event.target.files?.[0]
        if (!file) return
        const fileReader = new FileReader()
        await new Promise<void>((resolve) => {
            fileReader.onloadend = async (f: any) => {
                let bytes = new Uint8Array(f.target.result)
                const result = fileType(bytes)?.[0]
                const jpg = result?.mime === "image/jpeg" ||
                path.extname(file.name).toLowerCase() === ".jpg" ||
                path.extname(file.name).toLowerCase() === ".jpeg"
                const png = result?.mime === "image/png"
                const webp = result?.mime === "image/webp"
                if (jpg) result.typename = "jpg"
                if (jpg || png || webp) {
                    const url = functions.arrayBufferToBase64(bytes.buffer)
                    const link = `${url}#.${result.typename}`
                    removeImage()
                    setTimeout(() => {
                        setIPImage(link)
                    }, 100)
                }
                resolve()
            }
            fileReader.readAsArrayBuffer(file)
        })
        if (event.target) event.target.value = ""
    }

    const removeImage = (event?: any) => {
        event?.preventDefault()
        event?.stopPropagation()
        setIPImage("")
        setIPMaskImage("")
        setIPMaskData("")
        setImg(null)
        setMaskImg(null)
    }

    const loadImages = async () => {
        if (!ipImage) return
        const image = document.createElement("img")
        await new Promise<void>((resolve) => {
            image.onload = () => resolve()
            image.src = ipImage
        })
        setImg(image)
        if (!ipMaskImage) return
        const mask = document.createElement("img")
        await new Promise<void>(async (resolve) => {
            mask.onload = () => resolve()
            mask.src = ipMaskImage
        })
        setMaskImg(mask)
    }
    useEffect(() => {
        loadImages()
    }, [ipImage, ipMaskImage])

    const updateImage = () => {
        if (!ref.current || !img) return
        ref.current.width = functions.getNormalizedDimensions(img).width
        ref.current.height = functions.getNormalizedDimensions(img).height
        const ctx = ref.current.getContext("2d")!
        ctx.drawImage(img, 0, 0, ref.current.width, ref.current.height)

        if (maskImg) {
            const maskCanvas = document.createElement("canvas")
            maskCanvas.width = maskImg.width 
            maskCanvas.height = maskImg.height
            const maskCtx = maskCanvas.getContext("2d")!
            maskCtx.drawImage(maskImg, 0, 0, ref.current.width, ref.current.height)

            const imgData = ctx.getImageData(0, 0, ref.current.width, ref.current.height)
            const maskPixels = maskCtx.getImageData(0, 0, ref.current.width, ref.current.height).data

            const pixels = imgData.data
            for (let i = 0; i < pixels.length; i+=4) {
                if (maskPixels[i] < 10 && maskPixels[i+1] < 10 && maskPixels[i+2] < 10) {
                    // ignore
                } else {
                    const color = functions.rotateColor("#fc1594", siteHue, siteSaturation, siteLightness)
                    const {r, g, b} = functions.hexToRgb(color)
                    pixels[i] = r 
                    pixels[i+1] = g
                    pixels[i+2] = b
                }
            }
            ctx.putImageData(imgData, 0, 0)
        }
    }

    useEffect(() => {
        updateImage()
    }, [img, ipProcessor, maskImg, siteHue, siteSaturation, siteLightness])

    const startDrawing = (event: any) => {
        if (!ipImage) return
        event?.preventDefault()
        event?.stopPropagation()
        setIPDrawImage(ipImage)
    }

    const updateIPAdapterModels = async (first?: boolean) => {
        let subfolder = "models"
        if (modelName.includes("XL")) subfolder = "sdxl_models"
        const ipAdapterNames = await axios.get("/ip-adapter-models", {params: {subfolder}}).then((r) => r.data)
        setIPAdapterNames(ipAdapterNames)
        if (ipAdapter === "None") setIPAdapter(ipAdapterNames[0])
    }

    useEffect(() => {
        if (!ipAdapter) return
        setTimeout(() => {
            updateIPAdapterModels()
        }, 200)
    }, [ipAdapter, modelName])

    const ipModelsJSX = () => {
        let jsx = [] as any
        for (let i = 0; i < ipAdapterNames.length; i++) {
            jsx.push(<Dropdown.Item active={ipAdapter === ipAdapterNames[i]} onClick={() => setIPAdapter(ipAdapterNames[i])}>{ipAdapterNames[i]}</Dropdown.Item>)
        }
        return jsx 
    }

    const ipAdapterOptionsJSX = () => {
        return (
            <div className="control-image-options-container">
                <div className="control-option-row">
                    <span className="control-option-text" style={{marginRight: "10px"}}>Model</span>
                    <DropdownButton title={ipAdapter} drop="down" className="checkpoint-selector" onClick={() => updateIPAdapterModels()}>
                        {ipModelsJSX()}
                    </DropdownButton>
                </div>
                <div className="control-option-row">
                    <span className="control-option-text">Weight</span>
                    <Slider className="control-slider" trackClassName="control-slider-track" thumbClassName="control-slider-thumb" onChange={(value) => setIPWeight(value)} min={0} max={1} step={0.01} value={ipWeight}/>
                    <span className="control-option-text-value" style={{width: "60px"}}>{ipWeight}</span>
                </div>
            </div>
        )
    }

    return (
        <div className="controlnet" style={{alignItems: "flex-start"}}>
            <div className="controlnet-container" style={{marginLeft: "25px", width: "95%"}}>
                <div className="controlnet-title">IP Adapter</div>
                <div className="controlnet-buttons-container" style={{justifyContent: "flex-start"}}>
                    <div className="controlnet-button-container" onClick={() => setIPProcessor("off")}>
                        <img className="controlnet-radio-button" src={getRadioButton(ipProcessor === "off")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Off</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setIPProcessor("on")}>
                        <img className="controlnet-radio-button" src={getRadioButton(ipProcessor === "on")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">On</button>
                    </div>
                </div>
                {ipProcessor === "on" ? <div className="control-image-drawer">
                    <div className="control-image-drawer-container">
                        <label htmlFor="ip-img" className="control-image-container" onMouseEnter={() => setImageHover(true)} onMouseLeave={() => setImageHover(false)}>
                            <div className={`control-image-button-container ${img && imageHover ? "show-control-image-buttons" : ""}`}>
                                <img className="control-image-button" src={drawIcon} onClick={startDrawing} draggable={false} style={{marginRight: "5px", height: "17px"}}/>
                                <img className="control-image-button" src={xIcon} onClick={removeImage} draggable={false} style={{height: "17px"}}/>
                            </div>
                            {img ? 
                            <canvas ref={ref} className="control-image" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%) hue-rotate(${imageHue - 180}deg) saturate(${imageSaturation}%)`, maxWidth: "250px", maxHeight: "250px"}}></canvas> :
                            <img className="control-image" src={imgPlaceHolder} style={{filter: getFilter(), maxWidth: "250px", maxHeight: "250px"}} draggable={false}/>}
                        </label>
                        <input id="ip-img" type="file" onChange={(event) => loadImage(event)}/>
                        {ipAdapterOptionsJSX()}
                    </div>
                </div> : null}
            </div>
        </div>
    )
}

export default IPAdapter