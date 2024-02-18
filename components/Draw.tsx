import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, DrawImageContext, MaskImageContext,
MaskDataContext, ImageBrightnessContext, ImageContrastContext} from "../Context"
import functions from "../structures/Functions"
import CanvasDraw from "../structures/CanvasDraw"
import inpaintCheck from "../assets/icons/inpaint-check.png"
import inpaintX from "../assets/icons/inpaint-x.png"
import inpaintClear from "../assets/icons/inpaint-clear.png"
import inpaintUndo from "../assets/icons/inpaint-undo.png"
import inpaintErase from "../assets/icons/inpaint-erase.png"
import inpaintDraw from "../assets/icons/inpaint-draw.png"
import "./styles/draw.less"

let lastScrollY = window.scrollY

const Draw: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {drawImage, setDrawImage} = useContext(DrawImageContext)
    const {maskImage, setMaskImage} = useContext(MaskImageContext)
    const {maskData, setMaskData} = useContext(MaskDataContext)
    const [brushSize, setBrushSize] = useState(25)
    const [brushColor, setBrushColor] = useState("rgba(252, 21, 148, 0.5)")
    const [erasing, setErasing] = useState(false)
    const [img, setImg] = useState(null) as any
    const imageRef = useRef<HTMLCanvasElement>(null)
    const maskRef = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedImage = localStorage.getItem("maskImage")
        if (savedImage) setMaskImage(savedImage)
        const savedData = localStorage.getItem("maskData")
        if (savedData) setMaskData(savedData)
    }, [])

    useEffect(() => {
        localStorage.setItem("maskImage", maskImage)
        localStorage.setItem("maskData", maskData)
    }, [maskImage, maskData])

    const getNormalizedDimensions = () => {
        let greaterValue = img.width > img.height ? img.width : img.height
        const ratio = greaterValue / 800
        const width = Math.floor(img.width / ratio)
        const height = Math.floor(img.height / ratio)
        return {width, height}
    }

    const increaseBrushSize = () => {
        setBrushSize((prev: number) => {
            let newVal = prev + 1
            if (newVal > 100) newVal = 100
            return newVal
        })
    }

    const decreaseBrushSize = () => {
        setBrushSize((prev: number) => {
            let newVal = prev - 1
            if (newVal < 1) newVal = 1
            return newVal
        })
    }

    useEffect(() => {
        const handleWheel = (event: WheelEvent) => {
            // @ts-ignore
            const trackPad = event.wheelDeltaY ? event.wheelDeltaY === -3 * event.deltaY : event.deltaMode === 0
            if (event.deltaY < 0) {
                if (trackPad) {
                    increaseBrushSize()
                } else {
                    decreaseBrushSize()
                }
            } else {
                if (trackPad) {
                    decreaseBrushSize()
                } else {
                    increaseBrushSize()
                }
            }
        }
        const handleKey = (event: KeyboardEvent) => {
            if (event.key === "q") decreaseBrushSize()
            if (event.key === "w") increaseBrushSize()
            if (event.key === "b") draw()
            if (event.key === "e") erase()
        }
        window.addEventListener("wheel", handleWheel, false)
        window.addEventListener("keydown", handleKey, false)
        return () => {
            window.removeEventListener("wheel", handleWheel)
            window.removeEventListener("keydown", handleKey)
        }
    }, [])

    useEffect(() => {
        setBrushColor(functions.rotateColor("rgba(252, 21, 148, 1)", siteHue, siteSaturation, siteLightness))
    }, [siteHue, siteSaturation, siteLightness])

    const loadImg = async () => {
        if (!drawImage) return setImg(null)
        const img = document.createElement("img")
        await new Promise<void>((resolve) => {
            img.onload = () => resolve()
            img.src = drawImage
        })
        setImg(img)
    }

    const updateMask = async () => {
        if (!maskData || !maskRef.current) return
        const parsedData = JSON.parse(maskData)
        for (let i = 0; i < parsedData.lines.length; i++) {
            parsedData.lines[i].brushColor = functions.rotateColor("rgba(252, 21, 148, 1)", siteHue, siteSaturation, siteLightness)
        }
        // @ts-ignore
        maskRef.current.loadSaveData(JSON.stringify(parsedData), true)
    }

    useEffect(() => {
        loadImg()
        setTimeout(() => {
            setBrushSize(brushSize - 1)
            setTimeout(() => {
                updateMask()
            }, 100)
        }, 100)
    }, [drawImage, maskData, siteHue, siteLightness, siteSaturation])

    const draw = () => {
        setErasing(false)
    }

    const erase = () => {
        setErasing(true)
    }

    const clear = () => {
        if (!maskRef.current) return
        // @ts-ignore
        maskRef.current.clear()
    }

    const undo = () => {
        if (!maskRef.current) return
        // @ts-ignore
        maskRef.current.undo()
    }

    const close = () => {
        setDrawImage("")
        setImg(null)
    }

    const convertToWhite = async (data: string) => {
        const img = document.createElement("img")
        await new Promise<void>((resolve) => {
            img.onload = () => resolve()
            img.src = data
        })
        const canvas = document.createElement("canvas")
        canvas.width = img.width
        canvas.height = img.height
        const ctx = canvas.getContext("2d")!
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const pixels = imgData.data 
        for (let i = 0; i < pixels.length; i+=4) {
            if (pixels[i] < 10 && pixels[i+1] < 10 && pixels[i+2] < 10) {
                // ignore
            } else {
                pixels[i] = 255
                pixels[i+1] = 255
                pixels[i+2] = 255
                pixels[i+3] = 255
            }
        }
        ctx.putImageData(imgData, 0, 0)
        return canvas.toDataURL("image/png")
    }

    const save = async () => {
        if (!maskRef.current) return
        // @ts-ignore
        const image = maskRef.current.getDataURL("png", false, "#000000")
        const mask = await convertToWhite(image)
        // @ts-ignore
        const data = maskRef.current.getSaveData()
        setMaskImage(mask)
        setMaskData(data)
        close()
    }

    if (!drawImage || !img) return null

    const width = getNormalizedDimensions().width
    const height = getNormalizedDimensions().height

    return (
        <div className="draw">
            <div className="draw-img-container">
                <div className="draw-button-container">
                    <img className="draw-button" onClick={draw} src={inpaintDraw} style={{filter: getFilter()}} draggable={false}/>
                    <img className="draw-button" onClick={erase} src={inpaintErase} style={{filter: getFilter()}} draggable={false}/>
                    <img className="draw-button" onClick={undo} src={inpaintUndo} style={{filter: getFilter()}} draggable={false}/>
                    <img className="draw-button" onClick={clear} src={inpaintClear} style={{filter: getFilter()}} draggable={false}/>
                    <img className="draw-button" onClick={close} src={inpaintX} style={{filter: getFilter()}} draggable={false}/>
                    <img className="draw-button" onClick={save} src={inpaintCheck} style={{filter: getFilter()}} draggable={false}/>
                </div>
                <CanvasDraw
                    //@ts-ignore
                    ref={maskRef}
                    className="draw-img"
                    lazyRadius={0}
                    brushRadius={brushSize}
                    brushColor={brushColor}
                    catenaryColor="rgba(252, 21, 148, 0)"
                    hideGrid={true}
                    canvasWidth={width}
                    canvasHeight={height}
                    imgSrc={drawImage}
                    erase={erasing}
                    loadTimeOffset={0}
                    eraseColor="white"
                    style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}
                />
                {/* <canvas ref={imageRef} className="draw-img"></canvas> */}
            </div>
            
        </div>
    )
}

export default Draw